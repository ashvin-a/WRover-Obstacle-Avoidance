import numpy as np
import depthai as dai
import cv2


# converting from camera intrinsics data to cloud point
def backproject_depth(depth, step=1):
    """
    # depth: wxh array of depths
    # fx, fy, cx, cy: camera intrinsics
    #  returns array of xyz points
    """
    H, W = depth.shape
    vs = np.arange(0, H, step)
    us = np.arange(0, W, step)

    uu, vv = np.meshgrid(us, vs)

    Z = depth[vv, uu]  # depth values
    X = (uu - 605) * (Z / 563.33333)  # intrinsics
    Y = (vv - 360) * (Z / 563.33333)

    """
         u_flat = uu.reshape(-1)
         v_flat = uu.reshape(-1)
    """

    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return pts
    # pts, u_flat, v_flat (aarav should chech this part


# using 3 random points to create a plane
'''
Why Random Sampling Works
Imagine:
60% of  points are ground
40% are noise
Probability that 3 random points are all ground:
0.6 × 0.6 × 0.6 = 0.216
~21% chance.
That’s actually high.
So in 600 iterations:
600 × 0.216 ≈ 129 good ground-only samples (that's enough
Each of those will produce nearly the same plane.
And that plane will have thousands of inliers.
While a random rock plane will have maybe 20 inliers.
So the true plane wins by vote count.
'''

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


# https://numpy.org/doc/stable/user/basics.indexing.html
# this directly implies image[row,col] = image[v,u]
# justifies pixel_mask[v_flat[mask],u_flat[mask]] = True

def inliers_to_pixels(mask_inliers, v_flat, u_flat, image_shape):
    H, W = image_shape  # if depth.shape == (400, 600) H = 400 rows, 600 columns
    # empty pixel mask
    pixel_mask = np.zeros((H, W), dtype=bool)
    pixel_mask[v_flat[mask_inliers], u_flat[mask_inliers]] = True
    return pixel_mask


"""
Concrete mini-example
Sampled 5 pixels and made 5 points:
u_flat = [10, 20, 30, 40, 50]
v_flat = [ 5,  5,  6,  7,  7]

mask_inliers = [False, True, False, True, True]
Then:
u_in = [20, 40, 50]
v_in = [ 5,  7,  7]
So you mark:
(v=5, u=20) True
(v=7, u=40) True
(v=7, u=50) True
"""


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

    up_vector = np.array([0, 1, 0])  # world up direction
    cos_thresh = np.cos(np.deg2rad(angle_thresh_deg))  # orginally , angle_thresh_degor np.arccos
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

        if np.dot(n, up_vector) < 0:
            n = -n  # Flip the normal so it faces "up"
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
            # convert back to pixels back  X = (uu - cx) * (Z / fx)
            # intrinsics  Y = (vv - cy) * (Z / fy)
            # equivalent mask to what we would have in the 2D depth image.

    if n_best is None:
        return False, None, None, None

    return True, n_best, d_best, mask_best


def main(depth_full):
    pts = backproject_depth(depth_full)
    mask = ransac_plane(pts=pts)
    ground_mask = np.array(mask[3])
    ground_mask = ground_mask.reshape(-1, 1280)

    depth_full = cv2.normalize(depth_full, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_full = cv2.cvtColor(depth_full, cv2.COLOR_GRAY2BGR)

    depth_full[ground_mask] = (255, 0, 0)
    cv2.imshow("obstacle avoidance", depth_full)
    cv2.waitKey(1)


with dai.Pipeline() as pipeline:
    monoLeft = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    monoRight = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.ROBOTICS)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(1280, 720)

    config = stereo.initialConfig

    # Median filter to remove the salt n pepper type pixels
    config.postProcessing.median = dai.MedianFilter.KERNEL_5x5
    config.postProcessing.thresholdFilter.maxRange = 8000  # 8.0m

    config.setConfidenceThreshold(30)

    monoLeftOut = monoLeft.requestOutput((1280, 720))
    monoRightOut = monoRight.requestOutput((1280, 720))

    monoLeftOut.link(stereo.left)
    monoRightOut.link(stereo.right)

    rightOut = monoRightOut.createOutputQueue()
    stereoOut = stereo.depth.createOutputQueue()

    pipeline.start()
    while pipeline.isRunning():
        ## --- Depth Data Processing ---
        stereoFrame = stereoOut.get()

        assert stereoFrame.validateTransformations()

        # Get frame and convert to meters
        depth = stereoFrame.getCvFrame().astype(np.float32) / 1000.0

        # Call the processing function, now passing the heading
        main(depth_full=depth)

        # if cv2.waitKey(1) == ord('q'):
        #     break

    pipeline.stop()

cv2.destroyAllWindows()
## --- END: Pipeline and Device Loop ---
