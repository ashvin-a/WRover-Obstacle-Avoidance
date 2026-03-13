import numpy as np
import depthai as dai
import cv2

'''
Camera frame:

          Y (down)
          ↓
          |
          |
          O ─────────→ X (right)
         /
        /
       /
      Z (forward)
      -opencv style 
      robotics systems (ROS base_link style):

X → forward

Y → left

Z → up
                  Image plane

             (u=0,v=0) top-left
                   +----------------+
                   |                |
                   |       ●        |  ← (cx,cy)
                   |                |
                   +----------------+

                     Camera
                        O
                        
                        Each pixel (u,v) corresponds to a ray leaving O.
                        
                        one specific pixel:

Suppose:

u > cx  → pixel to the right
v > cy  → pixel lower

Then:

                      Z
                      ↑
                      |
                      |
                      O────────────→ X
                          \
                           \
                             .
                             (X,Y,Z)

That 3D point is:

X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = depth

So:

If u increases → X increases → point moves right

If v increases → Y increases → point moves downward

If depth increases → point moves farther forward
'''
# converting from camera intrinsics data to cloud point
#Takes a 2D depth image and turns it into a 3D point cloud.
def backproject_depth(depth, step=1):
    H, W = depth.shape
    vs = np.arange(0, H, step)
    us = np.arange(0, W, step)

    uu, vv = np.meshgrid(us, vs)

    Z = depth[vv, uu]
    X = (uu - 605) * (Z / 563.33333)
    Y = (vv - 360) * (Z / 563.33333)

    # --- ADVANCED FILTERING ---
    # 1. Z > 0.1 : Ignore completely invalid points
    # 2. Z < 4.0 : Ignore anything further than 4 meters away (stops wall snapping)
    # 3. vv > H // 3 : Ignore the top 33% of the image (stops ceiling/high wall snapping)
    valid = (Z > 0.1) & (Z < 3.5) & (vv > H // 3)

    Z = Z[valid]
    X = X[valid]
    Y = Y[valid]
    
    u_flat = uu[valid]  
    v_flat = vv[valid]  

    pts = np.stack([X, Y, Z], axis=-1)
    
    return pts, u_flat, v_flat


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

'''
start with 3 points in 3D
         r *
           ]
            ]
p *----------* q
B) Create two direction vectors on the plane
        r *
       / 
      /
p *----------* q
   v2        v1
   
   p = [x1, y1, z1]
q = [x2, y2, z2]
r = [x3, y3, z3]

    v1 = q - p
    v2 = r - p
    Builds two direction vectors that lie on the plane.

These are:

from p to q

from p to r
'''
#This function constructs the equation of a plane from 3 points.
def plane_from_points(p, q, r):
    # returns (normal_vector n, plane_offset d)
    # plane equation: n·x + d = 0
    v1 = q - p
    v2 = r - p

    n = np.cross(v1, v2)
    #Computes the length of the normal vector.
    norm = np.linalg.norm(n)
    '''
    If the normal is basically zero, the points are degenerate.

That usually means:

the points are nearly collinear

they do not define a stable plane
n is the normal

d is the offset
d = -np.dot(n, p)

Builds the plane equation:

n · x + d = 0

Using point p, since p lies on the plane:
    '''
    if norm < 1e-6:
        return None, None

    n = n / norm
    d = -np.dot(n, p)
    return n, d


# https://numpy.org/doc/stable/user/basics.indexing.html
# this directly implies image[row,col] = image[v,u]
# justifies pixel_mask[v_flat[mask],u_flat[mask]] = True
'''
mask_inliers: boolean array telling which 3D points are ground

u_flat, v_flat: the pixel positions associated with those 3D points

image_shape: (H, W)
'''

def inliers_to_pixels(mask_inliers, v_flat, u_flat, image_shape):
    H, W = image_shape  # if depth.shape == (400, 600) H = 400 rows, 600 columns
    # empty pixel mask
    pixel_mask = np.zeros((H, W), dtype=bool)
    #For every point classified as an inlier, mark the corresponding pixel as True.
    pixel_mask[v_flat[mask_inliers], u_flat[mask_inliers]] = True
    #This works because image indexing is [row, col] = [v, u].,return pixel_mask,Returns the 2D image mask.
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
def ransac_plane(pts, iters=500, dist_thresh=0.15,
                 angle_thresh_deg=40, target=None, tol=1.0, min_inliers=400):
    # pts: Nx3 array of 3D points
    # iters: number of trials
    # dist_thresh: max distance for an inlier
    # angle_thresh_deg: allowable deviation from UP direction (to best fit normal)
    # target: constraint on d (height)
    # tol: tolerance for d
    # min_inliers: minimum inlier count required
    """
       pts = Nx3 array of 3D points

       Each point is:

           p = [X, Y, Z]

       Point cloud idea:

              Z (forward)
                 ↑
                 |
           .     .      .
        .    .     .        .
     .      .      .    .
            .   .

       Many of these points belong to the ground plane.
       """

    N = pts.shape[0]
    if N < 3:
        return False, None, None, None

    """
        Define what "UP" means.

               Y (up)
                ↑
                |
                |
                O------→ X

        A ground plane should have a normal vector
        pointing roughly upward.
        """

    up_vector = np.array([0, 1, 0])  # world up direction
    """
        Convert angle threshold into cosine threshold.

        We want plane normals within angle_thresh_deg of UP.

            UP
            ↑
            |\
            | \
            |  \ allowed region
            |   \
            |____\________

        dot(n, up_vector) = cos(angle)
        """
    cos_thresh = np.cos(np.deg2rad(angle_thresh_deg))  # orginally , angle_thresh_degor np.arccos
    rng = np.random.default_rng()

    """
        These variables keep track of the best plane we find.
        """
    best_inliers = -1
    n_best = None
    d_best = None
    mask_best = None

    """
        RANSAC main loop

        Repeat many times:

            1) pick 3 random points
            2) create a plane
            3) count how many points agree with it
            4) keep the best plane
        """
    for _ in range(iters):

        """
                Pick 3 random points from the cloud.

                        point cloud

                     .     .      .
                  .      .   .       .
               .      .      .      .
                    ^   ^      ^
                    |   |       |
                   p1     p2     p3
                   
        """
        # pick 3 random points
        idx = rng.choice(N, size=3, replace=False)
        p1, p2, p3 = pts[idx]

        """
               Build two vectors on the plane.

                      r *
                       \
                        \
               p *-------* q

               v1 = q - p
               v2 = r - p
               """
        # compute plane
        v1 = p2 - p1
        v2 = p3 - p1
        """
                Cross product of those vectors gives the plane normal.

                v1 × v2 → n

                           n
                           ↑
                           |
                p *--------|----* q
                   \       |    /
                    \      |   /
                      \    |  /
                        \  | /
                          r
                """
        n = np.cross(v1, v2)
        norm = np.linalg.norm(n)

        """
               If norm is extremely small, the points are nearly collinear.

               Example:

               p *------*------* q

               This does not define a stable plane.
               """
        if norm < 1e-6:
            continue
        """
                Normalize the normal vector.

                Now |n| = 1

                This simplifies distance calculations later.
                """
        n = n / norm
        """
                A plane normal can point either direction:

                     ↑ n
                ----------------
                     ↓ -n

                We flip it so the normal always points upward.
                """
        if np.dot(n, up_vector) < 0:
            n = -n  # Flip the normal so it faces "up"
        # enforce "up" orientation
        """
                Enforce that the plane is roughly horizontal.

                If the plane is too tilted (like a wall),
                reject it.

                Example rejected plane:

                     |
                     |
                     |
                -----+-----

                Example accepted plane:

                -------------------------
                """
        if np.dot(n, up_vector) < cos_thresh:
            continue
        """
                Compute plane offset.

                Plane equation:

                    n · x + d = 0

                If p1 lies on the plane:

                    n · p1 + d = 0
                    d = -n · p1
                """
        d = -np.dot(n, p1)

        # height constraint
        #ExampleIf you know the ground is about 1 meter below the camera, restrict d to that region.
        if target is not None:
            if abs(d - target) > tol:
                continue

        # compute distance to plane: |n·x + d|
        """
               Compute distance of EVERY point to this plane.

               distance = |n · x + d|

               Visually:

                          .
                        .
               ------------------- plane
                     .
                  """

        dist = np.abs(pts @ n + d)
        """
                Mark inliers.

                Points close enough to the plane are considered supporters.
                """
        """
               Count how many points agree with this plane.
               """
        mask = dist < dist_thresh
        count = int(mask.sum())

        # update best if there is an inlier improvement

        """
                RANSAC principle:

                The correct plane will have MANY inliers.

                A random plane will have very few.
                """
        if count > best_inliers and count >= min_inliers:
            """
                        Save this as the best plane so far.
            """
            best_inliers = count
            n_best = n.copy()
            d_best = float(d)
            mask_best = mask.copy()
            """
                       mask_best tells which 3D points lie on the plane.

                       Later this mask will be reshaped back
                       into the image shape to mark ground pixels.
                       """
            # convert back to pixels back  X = (uu - cx) * (Z / fx)
            # intrinsics  Y = (vv - cy) * (Z / fy)
            # equivalent mask to what we would have in the 2D depth image.

    if n_best is None:
        return False, None, None, None

    return True, n_best, d_best, mask_best


def main(depth_full):
    # 1. Backproject (Downsample with step=10)
    pts_sub, u_sub, v_sub = backproject_depth(depth_full, step=10)
    
    # Check if we even have enough points to run RANSAC
    if len(pts_sub) < 100:
        print("❌ Not enough valid depth points to find a floor.")
        cv2.imshow("obstacle avoidance", depth_full)
        cv2.waitKey(1)
        return

    # 2. RUN RANSAC
    success, n_best, d_best, mask_best = ransac_plane(
        pts=pts_sub, 
        iters=400,              # Slightly more iterations for stability
        dist_thresh=0.08,       # Tightened to 8cm (stops merging floor with low obstacles)
        angle_thresh_deg=40,    # Tightened to 40 degrees (rejects vertical walls!)
        min_inliers=450         # Lowered slightly since we cropped the image
    )

    # Setup visualizer
    depth_vis = cv2.normalize(depth_full, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

    if success:
        pts_full, u_full, v_full = backproject_depth(depth_full, step=1)
        
        dist_full = np.abs(pts_full @ n_best + d_best)
        full_mask = dist_full < 0.08 
        
        ground_mask_2d = inliers_to_pixels(full_mask, v_full, u_full, depth_full.shape)
        depth_vis[ground_mask_2d] = (255, 0, 0)

        # --- DEBUG PRINT STATEMENTS ---
        # Calculate how tilted the detected plane is relative to the camera lens
        up_vector = np.array([0, 1, 0])
        angle_deg = np.degrees(np.arccos(np.clip(np.dot(n_best, up_vector), -1.0, 1.0)))
        
        # d_best is roughly the distance from the camera to the floor in meters
        print(f"✅ Floor Found | Inliers: {np.sum(full_mask):05d} | Tilt: {angle_deg:04.1f}° | Cam Height (d): {d_best:04.2f}m")
    else:
        cv2.putText(depth_vis, "No Ground Found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(f"❌ RANSAC Failed. Valid sample points: {len(pts_sub)}")

    cv2.imshow("obstacle avoidance", depth_vis)
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
    config.postProcessing.median = dai.MedianFilter.KERNEL_7x7
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