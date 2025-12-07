import sympy as sp

def rotation_matrix(axis, angle):
    c = sp.cos(angle)
    s = sp.sin(angle)
    if axis == 'x':
        return sp.Matrix([[1, 0, 0],
                          [0, c, s],
                          [0, -s, c]])
    elif axis == 'y':
        return sp.Matrix([[c, 0, -s],
                          [0, 1, 0],
                          [s, 0, c]])
    elif axis == 'z':
        return sp.Matrix([[c, s, 0],
                          [-s, c, 0],
                          [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'.")

def symbolic_rotation_matrix(order, angles):
    """
    order: string of axes, e.g. 'zyx'
    angles: list/tuple of sympy symbols, e.g. (phi, theta, psi)
    """
    if len(order) != 3 or len(angles) != 3:
        raise ValueError("Order and angles must have length 3.")
    R = sp.eye(3)
    for axis, angle in zip(order, angles):
        R = R * rotation_matrix(axis, angle)
    return sp.simplify(R)



