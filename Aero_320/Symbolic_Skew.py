import sympy as sp
import numpy as np
def skew_symbolic(a):
    """
    Returns the symbolic skew-symmetric matrix [a]_x such that:
        [a]_x * b = a × b
    where a is a 3x1 Sympy Matrix or vector.
    """
    ax, ay, az = a
    return sp.Matrix([
        [0, -az, ay],
        [az, 0, -ax],
        [-ay, ax, 0]
    ])

def cross_symbolic(a, b):
    """
    Computes the symbolic cross product a × b using the skew-symmetric matrix form.
    """
    return skew_symbolic(a) * b

# Example symbolic usage:
ax, ay, az, bx, by, bz = sp.symbols('a_x a_y a_z b_x b_y b_z', real=True)

w, x, y = sp.symbols('w x y', real=True)
c = sp.Matrix([w, x, y])

a = sp.Matrix([ax, ay, az])
b = sp.Matrix([bx, by, bz])


aNorm = sp.sqrt(ax**2 + ay**2 + az**2)

unit = sp.transpose(aNorm) * aNorm

sp.pprint(unit)  # Should print the symbolic norm



# Skew-symmetric matrix and cross product
a_skew = skew_symbolic(a)
cSkew = skew_symbolic(c)
a_cross_b = cross_symbolic(a, b)
cCrossB = cross_symbolic(c, b)



# Display results
sp.pprint(cSkew)
sp.pprint(cCrossB)

print("Skew-symmetric matrix [a]_x =")
sp.pprint(a_skew)
print("\na × b =")
sp.pprint(a_cross_b)