import numpy as np
from scipy.sparse import dia_matrix, diags
from functools import partial

class Mesh():

    def __init__(self, cells, dimensions=1):
        self.cells = cells
        self.max = cells**dimensions
        self.dimensions = range(dimensions)
        self.delta = 1.0/cells
        self.upper_neighbours = [pow(cells, i) for i in self.dimensions]
        self.boundary = [boundary(self, d) for d in self.dimensions]
        self.neighbour_mask = [neighbours(self, neighb, self.boundary[i])
                for i, neighb in enumerate(self.upper_neighbours)]

def boundary(mesh, dimension):
    """ compute boundary vector as function of dimension
        with 0 indicating a bc and 1 for a inner cell """
    n_conseq_cells = int(mesh.cells**(dimension))
    stride = int(mesh.cells**(dimension + 1) - n_conseq_cells)
    row = [1 for _ in range(stride)] + [0 for _ in range(n_conseq_cells)]
    return np.array(row*int(mesh.max/(stride + n_conseq_cells)))

def neighbours(mesh, neighb_distance, boundary):
    """ compute neighbours matrix as function of dimension and boundary """
    self = diags(np.ones(mesh.max)*(-1.0), 0)
    neighb = diags(boundary[:-neighb_distance], neighb_distance)
    return neighb + self

def discScheme(mesh, dim, weights, selector=None):
    """ compute the diagonal matrix of interpolation weights """
    return sum([abs(mesh.neighbour_mask[dim])*w for w in weights(selector)])

def cds(mesh, dim, field):
    def weights(_):
         yield 0.5
    return discScheme(mesh, dim, weights)

def grad_(mesh, interp, flux, field):
    """ return the discrete vector of δ(u,φ)/δx """
    return sum([(interp(mesh, dim, field) - (interp(mesh, dim, field)).T) * np.multiply(field, flux)
                for dim in mesh.dimensions])

def laplace_(mesh, interp, field, gamma):
    """ return the discrete vector of δ/δx(Γδ(φ)/δx) """

    def face(interp, neighb, field, gamma):
        """ returns a face value """
        return np.multiply(interp*gamma, neighb*field)

    def sumFaces(interp, neighb, field, gamma):
        """ returns the sum of the face values along one dimension """
        return (face(interp.T, neighb.T*(-1.0), field, gamma)
                - face(interp, neighb, field, gamma))

    return sum([sumFaces(interp(mesh, dim, field), mesh.neighbour_mask[dim], field, gamma)
                for dim in mesh.dimensions])

def uniScalarField(mesh, value):
    return np.ones(mesh.max) * value

def uniVectorField(mesh, values):
    return [np.ones(mesh.max) * value[dim] for dim in mesh.dimensions]

def ddt(rhs, val, deltaT=0.5):
    return val + (deltaT*rhs)

# def wrapper(func, *args, **kwargs):
#     def wrapped():
#         return func(*args, **kwargs)
#     return wrapped
