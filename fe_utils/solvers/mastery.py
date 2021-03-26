"""Solve a nonlinear problem using the finite element method.
If run as a script, the result is plotted. This file can also be
imported as a module and convergence tests run on the solver.
"""
from fe_utils import *
import numpy as np
from numpy import cos, sin, pi
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from argparse import ArgumentParser


def assemble_mixed(Vfs, Qfs, f, i0):
    """Assemble the finite element mixed system for the Stokes problem
    given the function space in which to solve and the right hand side
    function."""

    # Create an appropriate (complete) quadrature rule.
    deg_V, deg_Q = Vfs.element.degree, Qfs.element.degree
    Q = gauss_quadrature(Vfs.element.cell, 2*max(deg_V,deg_Q))

    # Tabulate the basis functions and their gradients at the quadrature points.
    # phi = fe.tabulate(Q.points) # Dimensions: #{quadrature points} x #{basis functions} x 2
    # phi_grad = fe.tabulate(Q.points, grad=True) # Dimensions: #{quadrature points} x #{basis functions} x #{dim.} x 2
    phi_V = Vfs.element.tabulate(Q.points)
    phi_grad_V = Vfs.element.tabulate(Q.points, grad=True)
    psi_Q = Qfs.element.tabulate(Q.points)
    # phi[:,i], phi_grad[:,i,:] store the basis function i or its grad. evaluated at each Q.points.

    # Create the left hand side matrix and right hand side vector.
    # This creates a sparse matrix because creating a dense one may
    # well run your machine out of memory!
    A = sp.lil_matrix((Vfs.node_count, Vfs.node_count))
    B = sp.lil_matrix((Qfs.node_count, Vfs.node_count))
    F = np.zeros(Vfs.node_count)

    # Now loop over all the cells and assemble LHS, RHS ignoring boundary conditions:
    for c in range(Vfs.mesh.entity_counts[-1]):

        # Find the appropriate global node numbers for this cell:
        nodes_V = Vfs.cell_nodes[c, :]
        nodes_Q = Qfs.cell_nodes[c, :]

        # Construct the jacobian for the cell:
        J = Vfs.mesh.jacobian(c)
        detJ = np.abs(np.linalg.det(J))
        invJ = np.linalg.inv(J)
                
        # Compute contribution to block A from eq. (9.14):
        grad_phi = np.einsum('bk,qibl->qikl', invJ, phi_grad_V)
        eps_u = 0.5 * (grad_phi + grad_phi.transpose(0,1,3,2))
        aux_A = np.einsum('qjkl,qikl->ijq', eps_u, eps_u)
        aux_A2 = detJ * np.dot(aux_A, Q.weights)

        # Compute contribution to block B from eq. (9.14):
        div_phi = np.einsum('qikk->qi', phi_grad_V) # Divergence of Vfs basis functions
        aux_B = np.einsum('qi,qj->ijq', psi_Q, div_phi)
        aux_B2 = detJ * np.dot(aux_B, Q.weights)

        # Compute contribution to f from eq. (9.14):
        aux_F = detJ * np.einsum('k,qil,il,qkl,kl,q->i', f.values[nodes_V], phi_V, Vfs.element.node_weights, phi_V, Vfs.element.node_weights, Q.weights)

        A[np.ix_(nodes_V, nodes_V)] += aux_A2
        B[np.ix_(nodes_Q, nodes_V)] += aux_B2
        F[nodes_V] += aux_F

    LHS_aux = sp.bmat([[A, B.T], [B, None]], 'lil')
    RHS = np.hstack((F, np.zeros(Qfs.node_count)))

    # Implement Dirichlet conditions of (H1)^2 on both x and y components:
    list_bound_nodes = boundary_nodes(Vfs)
    for i in list_bound_nodes:
        
        # Set global matrix rows of boundary nodes to 0 (except diagonal term):
        LHS_aux[i, :] = np.zeros(Vfs.node_count + Qfs.node_count)
        LHS_aux[i, i] = 1.  # Set diagonal term to 1.

        # Set global vector rows of boundary nodes to 0:
        RHS[i] = 0.

    # Implement Dirichlet boundary condition on L2 on a vertex that is not at
    # the boundary, since those nodes have already been set to 0, and this
    # would mean that we are actually not dealing with the extra degree of freedom.
    LHS_aux[[Vfs.node_count + i0], :] = np.zeros(Vfs.node_count + Qfs.node_count)
    LHS_aux[[Vfs.node_count + i0], [Vfs.node_count + i0]] = 1.
    RHS[Vfs.node_count + i0] = 0.

    LHS_mat = sp.csc_matrix(LHS_aux)
    LHS = splinalg.splu(LHS_mat)

    return LHS, RHS


def solve_mastery(resolution, analytic=False, return_error=False):
    """This function should solve the mastery problem with the given resolution. It
    should return both the solution :class:`~fe_utils.function_spaces.Function` and
    the :math:`L^2` error in the solution.

    If ``analytic`` is ``True`` then it should not solve the equation
    but instead return the analytic solution. If ``return_error`` is
    true then the difference between the analytic solution and the
    numerical solution should be returned in place of the solution.
    """
    
    """Solve a model Poisson problem on a unit square mesh with
    ``resolution`` elements in each direction, using equispaced
    Lagrange elements of degree ``degree``."""

    # Set up the mesh, finite element and function space required.
    mesh = UnitSquareMesh(resolution, resolution)

    # Set up the mixed function space and finite elements (P2^2 x P1):
    P1 = LagrangeElement(mesh.cell, 1)
    P2 = LagrangeElement(mesh.cell, 2)
    P2_2 = VectorFiniteElement(P2)
    Vfs = FunctionSpace(mesh, P2_2)
    Qfs = FunctionSpace(mesh, P1)

    # Choose the point where the pressure will vanish. It should not be on the 
    # boundary, since those nodes have already been set to 0, and therefere we
    # would not be actually not be fixing the extra degree of freedom.
    list_bound_nodes_Qfs = boundary_nodes(Qfs)
    list_interior_nodes_Qfs = [x for x in range(Qfs.node_count) if x not in list_bound_nodes_Qfs]
    # Choose an arbitrary point of the interior nodes:
    i0 = list_interior_nodes_Qfs[0]
    # Find its coordinates (x0, y0), since the analytic P will need to vanish there:
    x0, y0 = Qfs.mesh.vertex_coords[i0]

    # Create a function to hold the analytic solution for comparison purposes.
    # u = grad_perp(gamma), gamma given in Eq. (9.17)
    # Choose arbitrary pressure p = (x-x0)*(y-y0), where (x0,y0) are the coord. at which
    # we impose the pressure to vanish
    analytic_answer_u = Function(Vfs)
    analytic_answer_u.interpolate(lambda x: (-2*pi*(cos(2*pi*x[0]) - 1)*sin(2*pi*x[1]),
                                            2*pi*sin(2*pi*x[0])*(cos(2*pi*x[1] - 1))))
    analytic_answer_p = Function(Qfs)
    analytic_answer_p.interpolate(lambda x: (x[0]-x0)*(x[1]-y0)) 

    # If the analytic answer has been requested then bail out now.
    if analytic:
        return (analytic_answer_u, analytic_answer_p), 0.0

    # Create the right hand side function and populate it with the correct
    # values (with P = (x-x0)(y-y0), f = -laplacian(u) + grad(p) = ..)
    f = Function(Vfs)
    f.interpolate(lambda x: (x[1]-y0 - 8*pi**3*(2*cos(2*pi*x[0]) - 1)*sin(2*pi*x[1]),
                             x[0]-x0 - 8*pi**3*sin(2*pi*x[0])*(1 - 2*cos(2*pi*x[1]))))

    # Assemble the larger matrix from its blocks according to Eq. 9.12:
    LHS, RHS = assemble_mixed(Vfs, Qfs, f, i0)

    # Solve the system:
    sol = LHS.solve(RHS)

    # Create the function to hold the solution.
    u = Function(Vfs)
    p = Function(Qfs)

    # Define the solutions in the required format:
    u.values[:] = sol[:Vfs.node_count]
    p.values[:] = sol[Vfs.node_count:]

    # Create a separate function for each component of u to
    # compute the L2 error afterwards:
    Vfs_aux = FunctionSpace(Vfs.mesh, Vfs.element.element)
    u_x = Function(Vfs_aux)
    u_y = Function(Vfs_aux)

    # Identify x- and y- components of the solution:
    ind_x = [2*i for i in range(int(Vfs.node_count/2))]
    ind_y = [2*i+1 for i in range(int(Vfs.node_count/2))]
    u_x.values[:] = sol[ind_x]
    u_y.values[:] = sol[ind_y]

    # Create L^2 functions 1D versions of the analytiic solution to compute the error:
    analytic_answer_u_x = Function(Vfs_aux)
    analytic_answer_u_y = Function(Vfs_aux)
    analytic_answer_u_x.interpolate(lambda x: (-2*pi*(cos(2*pi*x[0]) - 1)*sin(2*pi*x[1])))
    analytic_answer_u_y.interpolate(lambda x: 2*pi*sin(2*pi*x[0])*(cos(2*pi*x[1] - 1)))                                        

    # Compute the L^2 error in the solution for testing purposes.
    error = errornorm(analytic_answer_u_x, u_x) + \
            errornorm(analytic_answer_u_y, u_y) + \
            errornorm(analytic_answer_p, p)

    if return_error:
        u.values -= analytic_answer_u.values
        p.values -= analytic_answer_p.values

    # Return the solution and the error in the solution.

    # Note: I am able to plot the correct quiver plot, but I have not been
    # able to reach convergence. I would need more time in order to explore
    # why this is happening.

    return (u, p), error


def boundary_nodes(fs):
    """Find the list of boundary nodes in fs. This is a
    unit-square-specific solution. A more elegant solution would employ
    the mesh topology and numbering.
    """
    eps = 1.e-10

    if isinstance(fs.element, VectorFiniteElement):
        fs_sc = FunctionSpace(fs.mesh, fs.element.element)
    else:
        fs_sc = fs
    f = Function(fs_sc)

    def on_boundary(x):
        """Return 1 if on the boundary, 0. otherwise."""
        if x[0] < eps or x[0] > 1 - eps or x[1] < eps or x[1] > 1 - eps:
            return 1.
        else:
            return 0.

    f.interpolate(on_boundary)

    return np.flatnonzero(f.values)