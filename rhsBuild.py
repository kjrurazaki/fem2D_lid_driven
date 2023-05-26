import numpy as np


def build_rhs(model):
    """
    Compute RHS giver f and g (from exact solution)
    """
    rhs = np.zeros((2 * model.lines_A + model.lines_B, 1))
    return rhs
