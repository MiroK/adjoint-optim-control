from dolfin import *
from dolfin_adjoint import *
import numpy as np


def normalize_angle(angle):
    '''Make angle in [-pi, pi]'''
    assert angle >= 0 

    if angle < pi: 
        return angle
    if angle < 2*pi:
        return -((2*pi)-angle)
        
    return normalize_angle(angle - 2*pi)


class JetBCValue(Expression):
    '''
    Value of this expression is a vector field v(x, y) = A(theta)*e_r
    where A is the amplitude function of the polar angle and e_r is radial
    unit vector. The field is modulated such that

    1) at theta = theta0 \pm width/2 A is 0
    2) \int_{J} v.n dl = Q

    Here theta0 is the (angular) position of the jet on the cylinder, width
    is its angular width and finaly Q is the desired flux thought the jet.
    All angles are in degrees.
    '''
    def __init__(self, radius, width, theta0, Q, **kwargs):
        assert width > 0 and radius > 0 # Sanity. Allow negative Q for suction
        theta0 = np.deg2rad(theta0)
        assert theta0 >= 0  # As coming from deg to rad

        self.radius = radius
        self.width = np.deg2rad(width)
        # From deg2rad it is possible that theta0 > pi. Below we habe atan2 so
        # shift to -pi, pi
        self.theta0 = normalize_angle(theta0)

        self.Q = Q

    def eval(self, values, x):
        A = self.amplitude(x)
        xC = 0.
        yC = 0.

        values[0] = A*(x[0] - xC)
        values[1] = A*(x[1] - yC)

    def amplitude(self, x):
        theta = np.arctan2(x[1], x[0])

        # NOTE: motivation for below is cos(pi*(theta0 \pm width)/w) = 0 to
        # smoothly join the no slip.
        scale = self.Q/(2.*self.width*self.radius**2/pi)

        if abs(theta - self.theta0) < np.deg2rad(self.width):
            return scale*cos(pi*(theta - self.theta0)/self.width)
        elif abs(theta - self.theta0 + pi) < np.deg2rad(self.width):
            return -scale * cos(pi * (theta - self.theta0) / self.width)
        else:
            return 0

    # This is a vector field in 2d
    def value_shape(self):
        return (2, )


class JetBCValueDerivative(Expression):
    def __init__(self, radius, width, theta0, **kwargs):
        assert width > 0 and radius > 0  # Sanity. Allow negative Q for suction
        theta0 = np.deg2rad(theta0)
        assert theta0 >= 0  # As coming from deg to rad

        self.radius = radius
        self.width = np.deg2rad(width)
        # From deg2rad it is possible that theta0 > pi. Below we habe atan2 so
        # shift to -pi, pi
        self.theta0 = normalize_angle(theta0)

    def eval(self, values, x):
        A = self.amplitude(x)
        xC = 0.
        yC = 0.

        values[0] = A * (x[0] - xC)
        values[1] = A * (x[1] - yC)

    def amplitude(self, x):
        theta = np.arctan2(x[1], x[0])

        # NOTE: motivation for below is cos(pi*(theta0 \pm width)/w) = 0 to
        # smoothly join the no slip.
        scale = 1.0 / (2. * self.width * self.radius ** 2 / pi)

        if abs(theta - self.theta0) < np.deg2rad(self.width):
            return scale*cos(pi*(theta - self.theta0)/self.width)
        elif abs(theta - self.theta0 + pi) < np.deg2rad(self.width):
            return -scale * cos(pi * (theta - self.theta0) / self.width)
        else:
            return 0

    # This is a vector field in 2d
    def value_shape(self):
        return (2,)


# ------------------------------------------------------------------------------


if __name__ == '__main__':
    from xii import EmbeddedMesh
    from generate_msh import generate_mesh
    from msh_convert import convert
    import os

    root = 'turek_2d'
    shift = 60
    positions = [90-shift, 270+shift]

    geometry_params = {
        'output': './turek_2d.geo',
        'length': 2.2,
        'front_distance': 0.05+0.15,
        'bottom_distance': 0.05+0.15,
        'coarse_distance': 0.5,
        'coarse_size': 0.1,
        'jet_radius': 0.05,
        'width': 0.41,
        'cylinder_size': 0.01,
        'box_size': 0.05,
        'jet_positions': positions,
        'jet_width': 10,
        'clscale': 0.25
    }


    h5_file = '.'.join([root, 'h5'])
    # Regenerate mesh?
    if True: 
        generate_mesh(geometry_params, template='./geometry_2d.template_geo')
        msh_file = '.'.join([root, 'msh'])
        assert os.path.exists(msh_file)

        convert(msh_file, h5_file)
        assert os.path.exists(h5_file)

    comm = mpi_comm_world()
    h5 = HDF5File(comm, h5_file, 'r')
    mesh = Mesh()
    h5.read(mesh, 'mesh', False)

    surfaces = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
    h5.read(surfaces, 'facet')

    cylinder_noslip_tag = 4
    first_jet = cylinder_noslip_tag+1
    njets = len(positions)

    width = geometry_params['jet_width']
    radius = geometry_params['jet_radius']


    W = VectorFunctionSpace(mesh, 'CG', 2)
    fW = Function(W)

    tagged_positions = zip(range(first_jet, first_jet + njets), positions)
    print(tagged_positions)
    k = 0
    for tag, theta0 in tagged_positions:
        k += 1
        cylinder = EmbeddedMesh(surfaces, markers=[tag])
        
        x, y = SpatialCoordinate(cylinder)
    
        # cylinder_surfaces = cylinder.marking_function
        
        V = VectorFunctionSpace(cylinder, 'CG', 2)

        v = JetBCValue(radius, width, theta0, Q=1*(-1)**k, degree=5)

        bc = DirichletBC(W, v, surfaces, tag)
        bc.apply(fW.vector())

        f = interpolate(v, V)
        # Outer normal of the cylinder
        n = as_vector((x, y))/Constant(radius)
        
        print(theta0, tag)#abs(assemble(dot(v, n)*dx) - 2*tag), assemble(1*dx(domain=mesh))
    
        # For visual check
        f.rename('f', '0')
        File('foo_%d.pvd' % tag) << f

    fW.rename('f', '0')
    File('f.pvd') << fW
