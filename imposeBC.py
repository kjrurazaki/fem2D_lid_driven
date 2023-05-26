import numpy as np

from matplotlib import pyplot as plt

import display_results


def imposeBC(model, method):
    """
    Impose BCs
    """
    stiffMat = model.stiffMat.copy()
    rhs = model.rhs
    # Impose Dirichlet
    (
        stiffMat,
        rhs,
        u_dir_velocity,
        u_dir_pressure,
        boundary_nodes_velocity,
    ) = impose_dirichlet(model, stiffMat, rhs)

    return (stiffMat, rhs, u_dir_velocity, u_dir_pressure, boundary_nodes_velocity)


def impose_dirichlet(model, stiffMat, rhs):
    """
    Impose Dirichlet boundary conditions by lifting function
    """
    u_dir_velocity = np.zeros((model.lines_A, 2))
    index_nodes = list(range(0, model.lines_A))
    boundary_nodes_velocity = []

    u_dir_pressure = np.zeros((model.lines_B, 1))

    # Dirichlet nodes for velocity
    for idir in range(0, model.element.NDir_velocity):
        iglob = model.element.DirNod_velocity[idir][0]
        index_nodes.remove(iglob - 1)
        boundary_nodes_velocity.append(iglob - 1)
        u_dir_velocity[iglob - 1, 0] = model.element.DirVal_velocity[idir, 0]
        u_dir_velocity[iglob - 1, 1] = model.element.DirVal_velocity[idir, 1]

    u_dir = np.concatenate(
        (u_dir_velocity[:, 0], u_dir_velocity[:, 1], u_dir_pressure[:, 0])
    )
    dir_value = np.matmul(stiffMat, u_dir)
    rhs -= dir_value.reshape(-1, 1)

    for idir in range(0, model.element.NDir_velocity):
        iglob = model.element.DirNod_velocity[idir][0]
        stiffMat[:, iglob - 1] = 0
        stiffMat[iglob - 1, :] = 0
        stiffMat[iglob - 1, iglob - 1] = 1

        stiffMat[:, iglob - 1 + model.lines_A] = 0
        stiffMat[iglob - 1 + model.lines_A, :] = 0
        stiffMat[iglob - 1 + model.lines_A, iglob - 1 + model.lines_A] = 1

        rhs[iglob - 1, :] = model.element.DirVal_velocity[idir, 0]
        rhs[iglob - 1 + model.lines_A, :] = model.element.DirVal_velocity[idir, 1]

    model.boundary_nodes_velocity = boundary_nodes_velocity
    return stiffMat, rhs, u_dir_velocity, u_dir_pressure, boundary_nodes_velocity
