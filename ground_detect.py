import numpy as np
# converting from camera intrinsics data to cloud point
def backproject_depth(depth, fx, fy, cx, cy, step=8):
    """
    # depth: wxh array of depths 
    # fx, fy, cx, cy: camera intrinsics
   #  returns array of xyz points

    H, W = depth.shape
    vs = np.arange(0, H, step)
    us = np.arange(0, W, step)

    uu, vv = np.meshgrid(us, vs)

    Z = depth[vv, uu]                  # depth values
    X = (uu - cx) * (Z / fx)           # intrinsics 
    Y = (vv - cy) * (Z / fy)

    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return pts


# using 3 random points to create a plane
def plane_from_points(p, q, r):
    # returns (normal_vector n, plane_offset d)
    # plane equation: n·x + d = 0
    v1 = q - p
    v2 = r - p

    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return None, None

    n = n / norm
    d = -np.dot(n, p)
    return n, d


# 3D plane fitting with ransac (random numbers as of now)
def ransac_plane(pts, iters=500, dist_thresh=0.05,
                 angle_thresh_deg=30, target=None, tol=1.0, min_inliers=50):
     # pts: Nx3 array of 3D points
    # iters: number of trials
    # dist_thresh: max distance for an inlier
    # angle_thresh_deg: allowable deviation from UP direction (to best fit normal)
    # target: constraint on d (height)
    # tol: tolerance for d
    # min_inliers: minimum inlier count required

    N = pts.shape[0]
    if N < 3:
        return False, None, None, None

    up_vector = np.array([0, 1, 0])     # world up direction
    cos_thresh = angle_thresh_deg #orginally np.cos(np.deg2rad(angle_thresh_deg)), or np.arccos
    rng = np.random.default_rng()

    best_inliers = -1
    n_best = None
    d_best = None
    mask_best = None

    for _ in range(iters):

        # pick 3 random points
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = pts[idx]

        # compute plane
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)

        if norm < 1e-6:
            continue

        n = n / norm

        # enforce "up" orientation
        if np.dot(n, up_vector) < cos_thresh:
            continue

        d = -np.dot(n, p1)

        # height constraint
        if target is not None:
            if abs(d - target) > tol:
                continue

        # compute distance to plane: |n·x + d|
        dist = np.abs(pts @ n + d)
        mask = dist < dist_thresh
        count = int(mask.sum())

        # update best if there is an inlier improvement
        if count > best_inliers and count >= min_inliers:
            best_inliers = count
            n_best = n.copy()
            d_best = float(d)
            mask_best = mask.copy() 
            #convert back to pixels back  X = (uu - cx) * (Z / fx)           
            # intrinsics  Y = (vv - cy) * (Z / fy)
            # equivalent mask to what we would have in the 2D depth image. 

    if n_best is None:
        return False, None, None, None

    return True, n_best, d_best, mask_best
