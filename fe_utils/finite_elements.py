# Cause division to always mean floating point division.
from __future__ import division
import numpy as np
from .reference_elements import ReferenceInterval, ReferenceTriangle
np.seterr(invalid='ignore', divide='ignore')


def lagrange_points(cell, degree):
    """Construct the locations of the equispaced Lagrange nodes on cell.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct nodes.

    :returns: a rank 2 :class:`~numpy.array` whose rows are the
        coordinates of the nodes.

    The implementation of this function is left as an :ref:`exercise
    <ex-lagrange-points>`.

    """

    if cell.dim == 1: # 1D case: equispaced points in [0, 1]
        return np.array([[i/degree] for i in range(degree+1)])
        
    elif cell.dim == 2: # 2D case: {(i/p, j/p) | i <= i+j <= p}
        return np.array([[i/degree, j/degree] for i in range(degree+1) for j in range(degree-i+1)])

    else:
        raise Exception("A cell of degree > 2 has been passed.")


def vandermonde_matrix(cell, degree, points, grad=False):
    """Construct the generalised Vandermonde matrix for polynomials of the
    specified degree on the cell provided.

    :param cell: the :class:`~.reference_elements.ReferenceCell`
    :param degree: the degree of polynomials for which to construct the matrix.
    :param points: a list of coordinate tuples corresponding to the points.
    :param grad: whether to evaluate the Vandermonde matrix or its gradient.

    :returns: the generalised :ref:`Vandermonde matrix <sec-vandermonde>`

    The implementation of this function is left as an :ref:`exercise
    <ex-vandermonde>`.
    """

    if grad:
        if cell.dim == 1: # Insert max(0, exp) in the exponents to avoid terms of the form 0*(x**(-1)) that give rise to NaN
            return np.array([[[j*points[k][0]**(max(0,j-1))] for j in range(degree+1)] for k in range(len(points))] )
        elif cell.dim == 2:
            return np.array([[[((i-j)*points[k][0]**(max(0,i-j-1)))*(points[k][1]**j), (points[k][0]**(i-j))*(j*points[k][1]**(max(0,j-1)))] for i in range(0, degree+1) for j in range(i+1)] for k in range(len(points))])

    else:
        if cell.dim == 1: # 1D case
            return np.array([[points[k][0]**j for j in range(degree+1)] for k in range(len(points))])
        elif cell.dim == 2: # 2D case
            return np.array([[(points[k][0]**(i-j))*(points[k][1]**j) for i in range(0, degree+1) for j in range(i+1)] for k in range(len(points))])

    # If we are at this point of the function, it means cell.dim != 1, 2.
    raise Exception("A cell of degree > 2 has been passed.")


class FiniteElement(object):
    def __init__(self, cell, degree, nodes, entity_nodes=None):
        """A finite element defined over cell.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.
        :param nodes: a list of coordinate tuples corresponding to
            the nodes of the element.
        :param entity_nodes: a dictionary of dictionaries such that
            entity_nodes[d][i] is the list of nodes associated with entity `(d, i)`.

        Most of the implementation of this class is left as exercises.
        """

        #: The :class:`~.reference_elements.ReferenceCell`
        #: over which the element is defined.
        self.cell = cell
        #: The polynomial degree of the element. We assume the element
        #: spans the complete polynomial space.
        self.degree = degree
        #: The list of coordinate tuples corresponding to the nodes of
        #: the element.
        self.nodes = nodes
        #: A dictionary of dictionaries such that ``entity_nodes[d][i]``
        #: is the list of nodes associated with entity `(d, i)`.
        self.entity_nodes = entity_nodes

        if entity_nodes:
            #: ``nodes_per_entity[d]`` is the number of entities
            #: associated with an entity of dimension d.
            self.nodes_per_entity = np.array([len(entity_nodes[d][0])
                                              for d in range(cell.dim+1)])

        # Replace this exception with some code which sets
        # self.basis_coefs
        # to an array of polynomial coefficients defining the basis functions.
        self.basis_coefs = np.linalg.inv(vandermonde_matrix(cell, degree, nodes))

        #: The number of nodes in this element.
        self.node_count = nodes.shape[0]

    def tabulate(self, points, grad=False):
        """Evaluate the basis functions of this finite element at the points
        provided.

        :param points: a list of coordinate tuples at which to
            tabulate the basis.
        :param grad: whether to return the tabulation of the basis or the
            tabulation of the gradient of the basis.

        :result: an array containing the value of each basis function
            at each point. If `grad` is `True`, the gradient vector of
            each basis vector at each point is returned as a rank 3
            array. The shape of the array is (points, nodes) if
            ``grad`` is ``False`` and (points, nodes, dim) if ``grad``
            is ``True``.

        The implementation of this method is left as an :ref:`exercise
        <ex-tabulate>`.

        """
        
        # The tabulation matrix is given by (V(X:)*C)_{ij}, where
        #  - C are the coeff. of the basis functions wrt monomial basis
        #  - V(X:) is the Vandermonde matrix ev. at the quadrature points
        # When grad=True, it is \nabla(V)·C, T_{ijk} = \nabla(\phi_j(X_i))·e_k

        V_X = vandermonde_matrix(self.cell, self.degree, points, grad=grad)
        if grad:
            return np.einsum("ijk,jl->ilk", V_X, self.basis_coefs)
        else:
            return np.dot(V_X, self.basis_coefs)


    def interpolate(self, fn):
        """Interpolate fn onto this finite element by evaluating it
        at each of the nodes.

        :param fn: A function ``fn(X)`` which takes a coordinate
           vector and returns a scalar value.

        :returns: A vector containing the value of ``fn`` at each node
           of this element.

        The implementation of this method is left as an :ref:`exercise
        <ex-interpolate>`.

        """

        raise NotImplementedError

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.cell,
                               self.degree)


class LagrangeElement(FiniteElement):
    def __init__(self, cell, degree):
        """An equispaced Lagrange finite element.

        :param cell: the :class:`~.reference_elements.ReferenceCell`
            over which the element is defined.
        :param degree: the
            polynomial degree of the element. We assume the element
            spans the complete polynomial space.

        The implementation of this class is left as an :ref:`exercise
        <ex-lagrange-element>`.
        """

        # Provide the nodes of the equispaced Lagrange elements:
        nodes = lagrange_points(cell, degree)
        
        # Use lagrange_points to obtain the set of nodes.  Once you
        # have obtained nodes, the following line will call the
        # __init__ method on the FiniteElement class to set up the
        # basis coefficients.
        super(LagrangeElement, self).__init__(cell, degree, nodes)
