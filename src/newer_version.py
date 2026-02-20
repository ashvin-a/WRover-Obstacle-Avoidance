import numpy as np
import depthai as dai
import cv2



class PlaneDetector:
    #https://web.stanford.edu/class/cs231a/course_notes/01-camera-models.pdf
    ## pinhole camera back-projection (pinhole camera model
    # converting from camera intrinsics data to cloud point
    # vprojection: u = f * X/Z + c_x (and similarly for v)
    # back-projection: X = (u - c_x) * Z / f, Y = (c_y - v) * Z / f
    def __init__(self, x_pixel_offset_cx=640.0, y_pixel_offset_cy=360.0, focal_length=563.33333):
        self.x_pixel_offset_cx = float(x_pixel_offset_cx) #x_pixel_offset_cx, y_pixel_offset_cy are cx,cy in pixel format
        self.y_pixel_offset_cy = float(y_pixel_offset_cy)
        self.focal_length = float(focal_length) #is effectively f (in pixels).

    def generate_point_cloud(self, depth_frame):
        """
        depth_frame: HxW depth array (meters recommended)
        Returns:
          points_valid: (M,3) valid 3D points
          valid_flat_idx: (M,) indices into flattened H*W pixels for each returned point
          H, W: original image shape
        """
        H, W = depth_frame.shape #reads image height/width from the depth frame.

        u = np.arange(W, dtype=np.float32)  #[0, 1, 2, ..., W-1]
        v = np.arange(H, dtype=np.float32) #[0..H-1]
        uu, vv = np.meshgrid(u, v) #https://numpy.org/devdocs/reference/generated/numpy.meshgrid.html

        '''
        uu[y, x] = x_pixel

        vv[y, x] = y_pixel
        '''


        z = depth_frame.astype(np.float32) # expands those into two full H×W grids:

        # These two lines implement pinhole back-projection using (c_x, c_y, f)
        #implement pinhole back-projection using (c_x, c_y, f) ( stored intrinsics).
        x = (uu - self.x_pixel_offset_cx) * z / self.focal_length
        y = (self.y_pixel_offset_cy - vv) * z / self.focal_length

        # https://numpy.org/doc/2.2/reference/generated/numpy.isfinite.html
        points = np.stack((x, y, z), axis=-1).reshape(-1, 3) #makes an H×W×3 array: last dimension is (x,y,z)

        valid = np.isfinite(points).all(axis=1) & (points[:, 2] > 0) #marks values that are not NaN, equires all 3 coords finite per point.
        valid_flat_idx = np.flatnonzero(valid)
        return points[valid], valid_flat_idx, H, W
    '''
    plane in 3D can be written as:ax+by+cz+d=0
    or in vector form:
    n⋅x+d=0
    n = (a,b,c) is the plane’s normal vector
    x = (x,y,z) is any point on the plane
    d shifts the plane
    need a normal vector and a point on the plane to define a plane
    # form two vectors lying inside the plane:
    
    '''

    def _fit_plane_from_3pts(self, p1, p2, p3, eps=1e-12):
        # v1=p2-p1=(x2-x1,y2-y1,z2-z1)
        # v2=p3-p1=(x3-x1,y3-y1,z3-z1)
        # If the points are nearly collinear, v1 and v2 are almost parallel.
        v1 = p2 - p1
        v2 = p3 - p1
        #cross product:n=v1×v2
        #The cross product of two vectors, produces a new vector that is orthogonal (perpendicular) to place with original vectors
        '''
                v2
               /
              /
        p1---/------> v1
        
        
        '''
        #https: // www.maplesoft.com / support / help / Maple / view.aspx?path = MathApps / EquationofaPlane3Points & cid = 959
        n = np.cross(v1, v2) #col i: v1x,v2x col j:v1y,v2y col k: v1z,v2z
        norm = np.linalg.norm(n) #||n||=root(a2 + b2 + c2)
        #Parallel vectors -> cross product is around  0.So norm around  0 means these 3 points cannot form a stable plane


       # Reject degenerate samples. RANSAC will try again with different points.

        if norm < eps:
            return None, None
        #Normalize the normal vector.||n||=1
        n = n / norm #Distance from point to plane = |n*x + d| if n is unit length
        d = -np.dot(n, p1) # want the plane equation: n*x+d=0, plug p1 and find d
        return n, d # we now have nx + d = 0 which defines the plaen

    '''
    Uses  plane equation: nx + d
    because n is normalized, |nx + d| is the perpendicular distance.
    '''

    def _plane_point_distances(self, points, n, d):
        return np.abs(points @ n + d)

    def _refine_plane_svd(self, inlier_points):
        centroid = inlier_points.mean(axis=0)
        X = inlier_points - centroid
        _, _, vh = np.linalg.svd(X, full_matrices=False)
        n = vh[-1]
        n = n / np.linalg.norm(n)
        d = -np.dot(n, centroid)
        return n, d

    def ransac_plane(self, points, distance_threshold=0.02, max_iterations=300, seed=None, min_inliers=1000):
        rng = np.random.default_rng(seed)
        N = points.shape[0]
        if N < 3:
            return None, None, None

        best_inlier_mask = None
        best_inlier_count = 0
        best_n, best_d = None, None

        for _ in range(max_iterations):
            idx = rng.choice(N, size=3, replace=False)
            p1, p2, p3 = points[idx]

            n, d = self._fit_plane_from_3pts(p1, p2, p3)
            if n is None:
                continue

            dist = self._plane_point_distances(points, n, d)
            inlier_mask = dist < distance_threshold
            count = int(inlier_mask.sum())

            if count > best_inlier_count:
                best_inlier_count = count
                best_inlier_mask = inlier_mask
                best_n, best_d = n, d

        if best_inlier_mask is None or best_inlier_count < min_inliers:
            return None, None, None

        refined_n, refined_d = self._refine_plane_svd(points[best_inlier_mask])
        return refined_n, refined_d, best_inlier_mask

    def detect_plane_from_depth(self, depth_frame, distance_threshold=0.02, max_iterations=300, seed=None, min_inliers=1000):
        """
        Returns:
          n, d,
          plane_points, non_plane_points,
          inlier_mask_points (aligned to points_valid),
          plane_pixel_mask (HxW aligned to depth_frame)
        """
        points_valid, valid_flat_idx, H, W = self.generate_point_cloud(depth_frame)

        n, d, inlier_mask_points = self.ransac_plane(
            points_valid,
            distance_threshold=distance_threshold,
            max_iterations=max_iterations,
            seed=seed,
            min_inliers=min_inliers,
        )
        if n is None:
            return None, None, None, None, None, None

        plane_points = points_valid[inlier_mask_points]
        non_plane_points = points_valid[~inlier_mask_points]

        plane_pixel_mask = np.zeros(H * W, dtype=bool)
        plane_pixel_mask[valid_flat_idx[inlier_mask_points]] = True
        plane_pixel_mask = plane_pixel_mask.reshape(H, W)

        return n, d, plane_points, non_plane_points, inlier_mask_points, plane_pixel_mask

    def main(self, depth_frame_meters, distance_threshold=0.02, max_iterations=300, min_inliers=1000):
        """
        Drop-in per-frame call for your DepthAI pipeline.
        depth_frame_meters: HxW float32 depth in meters (you already do /1000.0)
        Returns:
          plane_pixel_mask (HxW bool), n, d
        """
        n, d, _, _, _, plane_pixel_mask = self.detect_plane_from_depth(
            depth_frame_meters,
            distance_threshold=distance_threshold,
            max_iterations=max_iterations,
            seed=None,
            min_inliers=min_inliers,
        )
        return plane_pixel_mask, n, d
