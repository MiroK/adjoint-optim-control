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
        scale *= cos(pi*(theta - self.theta0)/self.width)
        if abs(theta - self.theta0) < self.width:
            return scale
        
        # Opposite surface
        if abs(theta - self.theta0 + pi) < self.width:
            return -scale

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
        scale = cos(pi*(theta - self.theta0)/self.width)
        
        if abs(theta - self.theta0) < self.width:
            return scale
        
        if abs(theta - self.theta0 + pi) < self.width:
            return -scale

        return 0

    # This is a vector field in 2d
    def value_shape(self):
        return (2, ) 
