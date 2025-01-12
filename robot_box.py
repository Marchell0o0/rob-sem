def solve_robot_hand_eye_with_fixed_camera(self, H_A, H_B):
    """
    Solve AX=YB calibration problem using Kronecker product method.
    
    Args:
        H_A (list of SE3): Robot end-effector poses in base frame
        H_B (list of SE3): ArUco marker poses in camera frame
    
    Returns:
        tuple: (gripper_to_flange as SE3, camera_to_base as SE3)
    """
    assert len(H_A) == len(H_B), "Number of poses must match"
    if len(H_A) < 3:
        raise ValueError(f"Need at least 3 poses, got {len(H_A)}")

    def kron(A, B):
        """Kronecker product implementation."""
        return np.kron(A, B)

    # Build linear equations for rotation part
    A_rot = []
    B_rot = []

    for A, B in zip(H_A, H_B):
        R_A = A.rotation.rot  # 3x3 rotation matrix
        R_B = B.rotation.rot  # 3x3 rotation matrix
        
        # Each pair of poses gives us 9 equations
        A_rot.append(kron(np.eye(3), R_A) - kron(R_B.T, np.eye(3)))
        B_rot.append(np.zeros(9))

    # Stack all equations
    A_rot = np.vstack(A_rot)  # Shape: (9n x 9)
    B_rot = np.hstack(B_rot).reshape(-1, 1)  # Shape: (9n x 1)

    # Solve for rotation (vectorized form)
    vec_R_X = np.linalg.lstsq(A_rot, B_rot, rcond=None)[0].flatten()
    R_X = vec_R_X.reshape(3, 3)

    # Project onto SO(3)
    U, _, Vt = np.linalg.svd(R_X)
    R_X = U @ Vt

    # Calculate rotation part of Y (camera to base)
    R_Y_list = []
    for A, B in zip(H_A, H_B):
        R_Y_candidate = A.rotation.rot @ R_X @ B.rotation.rot.T
        U, _, Vt = np.linalg.svd(R_Y_candidate)
        R_Y_list.append(U @ Vt)
    R_Y = np.mean(R_Y_list, axis=0)
    U, _, Vt = np.linalg.svd(R_Y)
    R_Y = U @ Vt

    # Solve for translation
    A_trans = []
    B_trans = []

    for A, B in zip(H_A, H_B):
        t_A = A.translation.reshape(3, 1)
        t_B = B.translation.reshape(3, 1)
        
        A_block = np.hstack([R_A, -np.eye(3)])  # Shape: (3 x 6)
        B_val = t_A - R_A @ R_X @ t_B  # Shape: (3 x 1)
        
        A_trans.append(A_block)
        B_trans.append(B_val)

    # Stack all equations
    A_trans = np.vstack(A_trans)  # Shape: (3n x 6)
    B_trans = np.vstack(B_trans)  # Shape: (3n x 1)

    # Solve for translations
    trans_solution = np.linalg.lstsq(A_trans, B_trans, rcond=None)[0]
    t_X = trans_solution[:3].flatten()
    t_Y = trans_solution[3:].flatten()

    # Create SE3 transforms
    gripper_to_flange = SE3(
        translation=t_X,
        rotation=SO3(rotation_matrix=R_X)
    )

    camera_to_base = SE3(
        translation=t_Y,
        rotation=SO3(rotation_matrix=R_Y)
    )

    return gripper_to_flange, camera_to_base 