#%% Part 2.4
import numpy as np
p1 = np.array([0.2, 0.3], dtype=float)
p2 = np.array([0.7, 0.5], dtype=float)

displacement = p2 - p1
distance = np.linalg.norm(displacement)

direction = displacement / distance

print("Displacement:", displacement)
print("Distance:", distance)
print("Unit direction:", direction)

#%% Part 3.4
import numpy as np
a = np.array([1.0, 2.0])
b = np.array([3.0, 4.0])

dot_product = np.dot(a, b)

cos_theta = dot_product / (
    np.linalg.norm(a) * np.linalg.norm(b)
)
cos_theta = np.clip(cos_theta, -1.0, 1.0)

angle_rad = np.arccos(cos_theta)
angle_deg = np.degrees(angle_rad)

projection = (np.dot(a, b) / np.dot(b, b)) * b

print("Dot product:", dot_product)
print("Angle (degrees):", angle_deg)
print("Projection of a onto b:", projection)

#%% Part 4.4
import numpy as np

r = np.array([0.5, 0.0, 0.0])
force = np.array([0.0, 10.0, 0.0])

torque = np.cross(r, force)

print("Torque:", torque)

#%% Part 5.3
import numpy as np

theta = np.deg2rad(90)

R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

v = np.array([1.0, 0.0])
v_rotated = R @ v

print(v_rotated)

#%% Part 7.2
import numpy as np

def forward_kinematics(theta1, theta2, L1, L2):
    """
    Compute the end-effector position of a 2-link planar arm.

    Parameters
    ----------
    theta1, theta2 : float
        Joint angles in radians.
    L1, L2 : float
        Link lengths in meters.

    Returns
    -------
    numpy.ndarray
        End-effector position [x, y] in meters.
    """
    p1 = np.array([
        L1 * np.cos(theta1),
        L1 * np.sin(theta1)
    ])

    p2 = np.array([
        L2 * np.cos(theta1 + theta2),
        L2 * np.sin(theta1 + theta2)
    ])

    return p1 + p2


L1 = 0.5
L2 = 0.3

theta1 = np.deg2rad(30)
theta2 = np.deg2rad(45)

position = forward_kinematics(
    theta1, theta2, L1, L2
)

print("End-effector position:", position)


#%% Part 9
import numpy as np

def rotation_matrix_2d(theta):
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s],
        [s,  c]
    ])


theta = np.deg2rad(37)
R = rotation_matrix_2d(theta)

v = np.array([0.6, 0.8])
v_rotated = R @ v

# 1. Rotation matrices preserve vector magnitude.
assert np.isclose(
    np.linalg.norm(v_rotated),
    np.linalg.norm(v)
)

# 2. R^T R = I.
assert np.allclose(
    R.T @ R,
    np.eye(2)
)

# 3. A proper rotation has determinant +1.
assert np.isclose(
    np.linalg.det(R),
    1.0
)

# 4. Rotating by theta and then by -theta
#    should recover the original vector.
R_inverse = rotation_matrix_2d(-theta)

assert np.allclose(
    R_inverse @ v_rotated,
    v
)

print("All rotation tests passed.")