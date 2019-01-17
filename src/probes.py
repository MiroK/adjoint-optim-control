import numpy as np
from mpi4py import MPI as py_mpi
from dolfin import *

try:
    from iufl import icompile
    from ufl.corealg.traversal import traverse_unique_terminals
    from iufl.operators import eigw

except ImportError:
    print('iUFL can be obtained from https://github.com/MiroK/ufl-interpreter')


class DragProbe(object):
    '''Integral proble of drag over the tagged mesh oriented exterior surface n.ds'''
    def __init__(self, mu, n, ds, tags, flow_dir=Constant((1, 0))):
        self.dim = flow_dir.ufl_shape[0]
        self.mu = mu
        self.n = n
        self.ds = ds
        self.tags = tags
        self.flow_dir = flow_dir

    def sample(self, u, p):
        '''Eval drag given the flow state'''
        # Stress
        sigma = 2*Constant(self.mu)*sym(grad(u)) - p*Identity(self.dim)
        # The drag form
        form = sum(dot(dot(sigma, self.n), self.flow_dir)*self.ds(i) for i in self.tags)

        return assemble(form)


class VelocityNormProbe(object):
    '''Integral proble of velocity norm over the tagged mesh exterior surface ds'''
    def __init__(self, ds, tags):
        self.ds = ds
        self.tags = tags

    def sample(self, u):
        '''Eval v.v*ds given the flow state'''
        form = sum(dot(u, u)*self.ds(i) for i in self.tags)
        return assemble(form)


class PenetratedDragProbe(object):
    '''Drag on a penetrated surface
    https://physics.stackexchange.com/questions/21404/strict-general-mathematical-definition-of-drag
    '''
    def __init__(self, rho, mu, n, ds, tags, flow_dir=Constant((1, 0))):
        self.dim = flow_dir.ufl_shape[0]
        self.mu = mu
        self.rho = rho
        self.n = n
        self.ds = ds
        self.tags = tags
        self.flow_dir = flow_dir

    def sample(self, u, p):
        '''Eval drag given the flow state'''
        mu, rho, n = self.mu, self.rho, self.n
        # Stress
        sigma = 2*Constant(mu)*sym(grad(u)) - p*Identity(self.dim)
        # The drag form
        form = sum(dot(-rho*dot(outer(u, u), n) + dot(sigma, n), self.flow_dir)*self.ds(i)
                   for i in self.tags)

        return assemble(form)



class PointProbe(object):
    '''Perform efficient evaluation of function u at fixed points'''
    def __init__(self, u, locations):
        # The idea here is that u(x) means: search for cell containing x,
        # evaluate the basis functions of that element at x, restrict
        # the coef vector of u to the cell. Of these 3 steps the first
        # two don't change. So we cache them

        # Locate each point
        mesh = u.function_space().mesh()
        limit = mesh.num_entities(mesh.topology().dim())
        bbox_tree = mesh.bounding_box_tree()
        # In parallel x might not be on process, the cell is None then
        cells_for_x = [None]*len(locations)
        for i, x in enumerate(locations):
            cell = bbox_tree.compute_first_entity_collision(Point(*x))
            from dolfin import info
            if -1 < cell < limit:
                cells_for_x[i] = cell
        
        V = u.function_space()
        element = V.dolfin_element()

        size = V.ufl_element().value_size()
        # Build the sampling matrix
        evals = []
        for x, cell in zip(locations, cells_for_x):
            # If we own the cell we alloc stuff and precompute basis matrix
            if cell is not None:
                basis_matrix = np.zeros(size*element.space_dimension())
                coefficients = np.zeros(element.space_dimension())

                cell = Cell(mesh, cell)
                vertex_coords, orientation = cell.get_vertex_coordinates(), cell.orientation()
                # Eval the basis once
                element.evaluate_basis_all(basis_matrix, x, vertex_coords, orientation)

                basis_matrix = basis_matrix.reshape((element.space_dimension(), size)).T
                # Make sure foo is bound to right objections
                def foo(u, c=coefficients, A=basis_matrix, elm=cell, vc=vertex_coords):
                    # Restrict for each call using the bound cell, vc ...
                    u.restrict(c, element, elm, vc, elm)
                    # A here is bound to the right basis_matrix
                    return np.dot(A, c)
            # Otherwise we use the value which plays nicely with MIN reduction
            else:
                foo = lambda u, size=size: (np.finfo(float).max)*np.ones(size)

            evals.append(foo)

        self.probes = evals
        # To get the correct sampling on all cpus we reduce the local samples across
        # cpus
        self.comm = V.mesh().mpi_comm().tompi4py()
        self.readings = np.zeros(size*len(locations), dtype=float)
        self.readings_local = np.zeros_like(self.readings)
        # Return the value in the shape of vector/matrix
        self.nprobes = len(locations)
            
    def sample(self, u):
        '''Evaluate the probes listing the time as t'''
        self.readings_local[:] = np.hstack([f(u) for f in self.probes])    # Get local
        self.comm.Allreduce(self.readings_local, self.readings, op=py_mpi.MIN)  # Sync
        
        return self.readings.reshape((self.nprobes, -1))


class ExpressionProbe(object):
    '''Point evaluation of arbitrary scalar expressions'''
    # NOTE: no prior cell search, i.e. is slower
    #       does not work in parallel
    def __init__(self, expr, locations, mesh=None):
        # Extract mesh from one of the arguments
        if mesh is None:
            for arg in traverse_unique_terminals(expr):
                print arg
                if isinstance(arg, Function):
                    mesh = arg.function_space().mesh()
                    break
        assert mesh is not None

        expr = icompile(expr)
        size = expr.ufl_element().value_size()

        limit = mesh.num_entities(mesh.topology().dim())
        bbox_tree = mesh.bounding_box_tree()
        
        evals = []
        for x in locations:
            cell = bbox_tree.compute_first_entity_collision(Point(*x))

            if -1 < cell < limit:
                foo = lambda x=x, expr=expr: (expr)(x)
            else:
                foo = lambda size=size: (np.finfo(float).max)*np.ones(size)
            evals.append(foo)

        self.probes = evals
        # Return the value in the shape of vector/matrix
        self.nprobes = len(locations)
            
    def sample(self):
        '''Evaluate the probes listing the time as t'''
        # No args as everything is expr and thus is wired up to u_, p_
        readings = np.array([f() for f in self.probes])    # Get local        
        return readings.reshape((self.nprobes, -1))

# To make life easier we subclass each of the probes above to be able to init by 
# a FlowSolver instance and also unify the sample method to be called with both 
# velocity and pressure

class DragProbeANN(DragProbe):
    '''Drag on the cylinder'''
    def __init__(self, flow, flow_dir=Constant((1, 0))):
        DragProbe.__init__(self,
                           mu=flow.viscosity,
                           n=flow.normal,
                           ds=flow.ext_surface_measure,
                           tags=flow.cylinder_surface_tags,
                           flow_dir=flow_dir)

        
class LiftProbeANN(DragProbeANN):
    '''Lift on the cylinder'''
    def __init__(self, flow, flow_dir=Constant((0, 1))):
        DragProbeANN.__init__(self, flow, flow_dir)

        
class VelocityNormProbeANN(VelocityNormProbe):
    '''Velocity on the cylinder'''
    def __init__(self, flow):
        VelocityNormProbe.__init__(self,
                                   ds=flow.ext_surface_measure,
                                   tags=flow.cylinder_surface_tags)
                    
    def sample(self, u, p): return VelocityNormProbe.sample(self, u)                   


class PressureProbeANN(PointProbe):
    '''Point value of pressure at locations'''
    def __init__(self, flow, locations):
        PointProbe.__init__(self, flow.p_, locations)
    
    def sample(self, u, p): return PointProbe.sample(self, p)


class VelocityProbeANN(PointProbe):
    '''Point value of velocity vector at locations'''
    def __init__(self, flow, locations):
        PointProbe.__init__(self, flow.u_, locations)
    
    def sample(self, u, p): return PointProbe.sample(self, u)


class PenetratedDragProbeANN(PenetratedDragProbe):
    '''Drag on a penetrated surface
    https://physics.stackexchange.com/questions/21404/strict-general-mathematical-definition-of-drag
    '''
    def __init__(self, flow, flow_dir=Constant((1, 0))):
        PenetratedDragProbe.__init__(self,
                                     rho=flow.density,
                                     mu=flow.viscosity,
                                     n=flow.normal,
                                     ds=flow.ext_surface_measure,
                                     tags=flow.cylinder_surface_tags,
                                     flow_dir=flow_dir)


class StressEigwProbeANN(ExpressionProbe):
    '''Sample eigenvalues of a fluid stress'''
    def __init__(self, flow, locations):
        mu = flow.viscosity
        expr = eigw(-flow.p_*Identity(2) + 2*Constant(mu)*sym(grad(flow.u_)))

        mesh = flow.u_.function_space().mesh()

        ExpressionProbe.__init__(self, expr, locations, mesh=mesh)

    def sample(self, u, p): return ExpressionProbe.sample(self)


# ------------------------------------------------------------------------------


if __name__ == '__main__':
    from dolfin import *
    mesh = UnitSquareMesh(64, 64)

    # #########################
    # # Check scalar 
    # #########################
    V = FunctionSpace(mesh, 'CG', 2)
    f = Expression('t*(x[0]+x[1])', t=0, degree=1)
    # NOTE: f(x) has issues in parallel so we don't do f eval
    # through fenics
    f_ = lambda t, x: t*(x[:, 0]+x[:, 1])
    
    u = interpolate(f, V)
    locations = np.array([[0.2, 0.2],
                          [0.8, 0.8],
                          [1.0, 1.0],
                          [0.5, 0.5]])

    probes = PointProbe(u, locations)

    for t in [0.1, 0.2, 0.3, 0.4]:
        f.t = t
        u.assign(interpolate(f, V))
        # Sample f
        ans = probes.sample(u)
        truth = f_(t, locations).reshape((len(locations), -1))
        # NOTE: that the sample always return as matrix, in particular
        # for scale the is npoints x 1 matrix
        assert np.linalg.norm(ans - truth) < 1E-14, (ans, truth)

    # ##########################
    # # Check vector
    # ##########################
    V = VectorFunctionSpace(mesh, 'CG', 2)
    f = Expression(('t*(x[0]+x[1])',
                    't*x[0]*x[1]'), t=0, degree=2)
    # NOTE: f(x) has issues in parallel so we don't do f eval
    # through fenics
    f0_ = lambda t, x: t*(x[:, 0]+x[:, 1])
    f1_ = lambda t, x: t*x[:, 0]*x[:, 1]

    u = interpolate(f, V)
    locations = np.array([[0.2, 0.2],
                          [0.8, 0.8],
                          [1.0, 1.0],
                          [0.5, 0.5]])

    probes = PointProbe(u, locations)

    for t in [0.1, 0.2, 0.3, 0.4]:
        f.t = t
        u.assign(interpolate(f, V))
        # Sample f
        ans = probes.sample(u)
        truth = np.c_[f0_(t, locations).reshape((len(locations), -1)),
                      f1_(t, locations).reshape((len(locations), -1))]

        assert np.linalg.norm(ans - truth) < 1E-14, (ans, truth)

    ##########################
    # Check expression
    ##########################
    V = VectorFunctionSpace(mesh, 'CG', 2)
    f = Expression(('t*(x[0]+x[1])',
                    't*(2*x[0] - x[1])'), t=0.0, degree=2)
    # NOTE: f(x) has issues in parallel so we don't do f eval
    # through fenics
    f0_ = lambda t, x: t*(x[:, 0]+x[:, 1])
    f1_ = lambda t, x: t*(2*x[:, 0]-x[:, 1])

    f_ = lambda t, x: f0_(t, x)**2 + f1_(t, x)**2

    u = interpolate(f, V)
    locations = np.array([[0.2, 0.2],
                          [0.8, 0.8],
                          [1.0, 1.0],
                          [0.5, 0.5]])

    probes = ExpressionProbe(inner(u, u), locations)

    for t in [0.1, 0.2, 0.3, 0.4]:
        f.t = t
        u.assign(interpolate(f, V))
        # Sample f
        ans = probes.sample(u)
        truth = f_(t, locations).reshape((len(locations), -1))

        assert np.linalg.norm(ans - truth) < 1E-14, (t, ans, truth)
        
    # Now for something fancy
    from iufl.operators import eigw
    # Eigenvalues of outer product of velocity
    probes = ExpressionProbe(eigw(outer(u, u)), locations, mesh=mesh)

    for t in [0.1, 0.2, 0.3, 0.4]:
        f.t = t
        u.assign(interpolate(f, V))
        # Sample f
        ans = probes.sample(u)
        print ans
