"""Solve a model helmholtz problem using the finite element method.
If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from fe_utils import *
import numpy as np
from numpy import cos, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser


def assemble(fs, f):
    """Assemble the finite element system for the Helmholtz problem given
    the function space in which to solve and the right hand side
    function."""

    # Create an appropriate (complete) quadrature rule.
    fe = fs.element
    Q = gauss_quadrature(fe.cell, fe.degree)

    # Tabulate the basis functions and their gradients at the quadrature points.
    phi = fe.tabulate(Q.points) # Dimensions: #{quadrature points} x #{basis functions}
    phi_grad = fe.tabulate(Q.points, grad=True) # Dimensions: #{quadrature points} x #{basis functions} x #{dim.}
    # phi[:,i], phi_grad[:,i,:] store the basis function i evaluated at each Q.points.
                  
    # Create the left hand side matrix and right hand side vector.
    # This creates a sparse matrix because creating a dense one may
    # well run your machine out of memory!
    A = sp.lil_matrix((fs.node_count, fs.node_count))
    l = np.zeros(fs.node_count)

    # Now loop over all the cells and assemble A and l:
    for c in range(fs.mesh.entity_counts[-1]):

        # Find the appropriate global node numbers for this cell:
        nodes = fs.cell_nodes[c, :]
        # Construct the jacobian for the cell:
        J = fs.mesh.jacobian(c)
        detJ = np.abs(np.linalg.det(J))
        invJT = np.linalg.inv(J.T)

        # Implement products in equation (6.72):
        v = np.array([np.dot( phi[:,i] * Q.weights, np.dot( f.values[nodes], phi.T ) ) * detJ for i in range(len(nodes))])
                
        # Equation (6.78):
        m = np.zeros((len(nodes), len(nodes)))
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                aux = np.array([np.dot( invJT @ phi_grad[q,i,:], invJT @ phi_grad[q,j,:] ) + phi[q,i]*phi[q,j] for q in range(len(Q.points))])
                m[i, j] = np.dot(aux, Q.weights) * detJ

        A[np.ix_(nodes, nodes)] += m
        l[nodes] += v

    # It seems like something is wrong...
    # test_11_helmholtz_convergence.py never finishes running
    return A, l


def solve_helmholtz(degree, resolution, analytic=False, return_error=False):
    """Solve a model Helmholtz problem on a unit square mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree ``degree``."""

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)
    fe = LagrangeElement(mesh.cell, degree)
    fs = FunctionSpace(mesh, fe)

    # Create a function to hold the analytic solution for comparison purposes.
    analytic_answer = Function(fs)
    analytic_answer.interpolate(lambda x: cos(4*pi*x[0])*x[1]**2*(1.-x[1])**2)

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return analytic_answer, 0.0

    # Create the right hand side function and populate it with the
    # correct values.
    f = Function(fs)
    f.interpolate(lambda x: ((16*pi**2 + 1)*(x[1] - 1)**2*x[1]**2 - 12*x[1]**2 + 12*x[1] - 2) *
                  cos(4*pi*x[0]))

    # Assemble the finite element system.
    A, l = assemble(fs, f)

    # Create the function to hold the solution.
    u = Function(fs)

    # Cast the matrix to a sparse format and use a sparse solver for
    # the linear system. This is vastly faster than the dense
    # alternative.
    A = sp.csr_matrix(A)
    u.values[:] = splinalg.spsolve(A, l)

    # Compute the L^2 error in the solution for testing purposes.
    error = errornorm(analytic_answer, u)

    if return_error:
        u.values -= analytic_answer.values

    # Return the solution and the error in the solution.
    return u, error

if __name__ == "__main__":

    parser = ArgumentParser(
        description="""Solve a Helmholtz problem on the unit square.""")
    parser.add_argument("--analytic", action="store_true",
                        help="Plot the analytic solution instead of solving the finite element problem.")
    parser.add_argument("--error", action="store_true",
                        help="Plot the error instead of the solution.")
    parser.add_argument("resolution", type=int, nargs=1,
                        help="The number of cells in each direction on the mesh.")
    parser.add_argument("degree", type=int, nargs=1,
                        help="The degree of the polynomial basis for the function space.")
    args = parser.parse_args()
    resolution = args.resolution[0]
    degree = args.degree[0]
    analytic = args.analytic
    plot_error = args.error

    u, error = solve_helmholtz(degree, resolution, analytic, plot_error)

    u.plot()
