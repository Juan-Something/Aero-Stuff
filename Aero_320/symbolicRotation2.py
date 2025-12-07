import sympy as sp


theta_x, theta_y, theta_z = sp.symbols('theta_x theta_y theta_z', real=True)

# -----------------------------
# Rotation matrix about a single axis
# -----------------------------
def rot(axis, angle):
    """Return 3x3 rotation matrix about axis 'x', 'y', or 'z'."""
    c = sp.cos(angle)
    s = sp.sin(angle)
    if axis in ('x', 'X', '1'):
        return sp.Matrix([
            [1, 0, 0],
            [0, c, s],
            [0, -s,  c]
        ])
    elif axis in ('y', 'Y', '2'):
        return sp.Matrix([
            [c, 0, -s],
            [0, 1, 0],
            [s, 0, c]
        ])
    elif axis in ('z', 'Z', '3'):
        return sp.Matrix([
            [c, s, 0],
            [-s,  c, 0],
            [0,  0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x','y','z' or '1','2','3'.")

# -----------------------------
# Build symbolic Euler rotation matrix
# -----------------------------
def euler_rotation_matrix(sequence='321'):
    """
    Return standardized intrinsic Euler rotation matrix (body-to-inertial, right-hand).
    sequence: string of 3 characters, e.g. '321', '231', 'ZYX'
    """
    seq = list(sequence)
    if len(seq) != 3:
        raise ValueError("Euler sequence must have 3 characters.")

    # Map theta symbols to rotation axes
    angle_map = {'x': theta_x, 'y': theta_y, 'z': theta_z,
                 '1': theta_x, '2': theta_y, '3': theta_z,
                 'X': theta_x, 'Y': theta_y, 'Z': theta_z}

    # Intrinsic rotation: last rotation * middle * first
    R1 = rot(seq[0], angle_map[seq[0]])
    R2 = rot(seq[1], angle_map[seq[1]])
    R3 = rot(seq[2], angle_map[seq[2]])
    
    C = sp.simplify(R3 * R2 * R1)  # intrinsic (body-to-inertial)
    return C

# Example usage
# -----------------------------
if __name__ == "__main__":
    seq = '231'  # change to '321', '312', etc.
    C = euler_rotation_matrix(seq)

    sp.pprint(C)

thetaY = sp.atan2(C[1,2], C[1,1])
sp.pprint(C[1,2])