import csv
import math
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from cv2.typing import MatLike
from scipy.spatial.transform import Rotation as R


class IncrementalHomography:
    """Class that implements incremental homography estimation"""

    def __init__(self, window_size: int, image0: MatLike, Rt0w: np.ndarray) -> None:
        """Constructor

        Args:
            window_size (int): number of images between homography is recalculated
            image0 (MatLike): initial image
            Rt0w (np.ndarray): SE(3) transform from image to world
        """
        self.window_size = window_size
        self.current_id = 0
        self.anchor_image = image0
        self.anchor_pose = Rt0w
        self.anchor_homography = np.eye(3)

    def calculate_incremental_homography(
        self, image: MatLike, Rtcw: np.ndarray, K: np.ndarray, filtered: bool, triangulation_threshold: bool
    ) -> np.ndarray:
        """Calculates honography transformation between image and image0 passed in the constructor.
        Homography is calculated as composition of newly estimated homography and previously saved homography.
        This enables incremental homography estimation,
        which is useful for wide baselines with many feature matching outliers.

        Args:
            image (MatLike): new image to calculate homography for
            Rtcw (np.ndarray): SE(3) transformation for new image
            K (np.ndarray): camera matrix
            filtered (bool): whether to use epipolar filtering
            triangulation_threshold (bool): whether to threshold matches by triangulated depths

        Returns:
            np.ndarray: estimated homography of new image compared to the initial image assigned in the constructor
        """
        H = get_homography_matching(
            self.anchor_image,
            image,
            self.anchor_pose,
            Rtcw,
            K,
            epipolar_threshold=filtered,
            triangulation_theshold=triangulation_threshold,
        )

        self.current_id += 1
        if self.current_id % self.window_size == 0:
            self.anchor_image = image
            self.anchor_pose = Rtcw
            self.anchor_homography = self.anchor_homography @ H
            return self.anchor_homography

        return self.anchor_homography @ H


def detect_match_features(
    image1: MatLike, image2: MatLike, n_features: int = 3000, n_matches: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:
    """Detect and match orb features

    Args:
        image1 (MatLike): first image
        image2 (MatLike): second image
        n_features (int, optional): number of feature detections. Defaults to 3000.
        n_matches (Optional[int], optional): number of strongest matches to keep. Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[DMatch]]: n_matches X 2 array of keypoint coordinates in image1 and image2,
                                                     list of matches
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create(n_features)

    # Find the keypoints and descriptors with ORB
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    # Create matcher object
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = matcher.match(descriptors1, descriptors2)
    matches = list(matches)

    # Sort matches by score
    matches.sort(key=lambda x: x.distance, reverse=False)

    # Remove not so good matches
    if n_matches is not None:
        matches = matches[:n_matches]

    return keypoints1, keypoints2, matches


def filter_matches(
    points1: np.ndarray, points2: np.ndarray, matches: List[cv2.DMatch], F: np.ndarray, threshold: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]:
    """Filter matches according to the epipolar constraing

    Args:
        points1 (np.ndarray): keypoint coordinates in ifirst image
        points2 (np.ndarray): keypoint coordinates in second image
        matches (List[cv2.DMatch]): matches
        F (np.ndarray): fundamental matrix
        threshold (int, optional): max distance from epipolar line. Defaults to 2.0.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[cv2.DMatch]]: _description_
    """
    epipolar_lines = cv2.computeCorrespondEpilines(np.array([p for p in points1]), 1, F).reshape(-1, 3)

    valid = np.array([distance_from_line(points2[i], epipolar_lines[i]) for i in range(len(matches))]) < threshold

    points1_filtered = points1[valid]
    points2_filtered = points2[valid]
    matches_filtered = [matches[i] for i in range(len(valid)) if valid[i]]

    return points1_filtered, points2_filtered, matches_filtered


def DLT(P1: np.ndarray, P2: np.ndarray, point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """Calculates triangulation Direct Lienar Transform

    Args:
        P1 (np.ndarray): projection matrix (camera matrix x SE(3) matrix) for first image
        P2 (np.ndarray): projection matrix for second image
        point1 (np.ndarray): coordinates of match in first image
        point2 (np.ndarray): coordinates of match in second image

    Returns:
        np.ndarray: triangulated 3D point
    """
    A = [
        point1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point1[0] * P1[2, :],
        point2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point2[0] * P2[2, :],
    ]
    A = np.array(A).reshape((4, 4))
    B = A.transpose() @ A  # type: ignore

    _, _, Vh = np.linalg.svd(B, full_matrices=False)

    return Vh[3, 0:3] / Vh[3, 3]


def triangulation(
    K: np.ndarray, Rtw1: np.ndarray, Rtw2: np.ndarray, points1: np.ndarray, points2: np.ndarray
) -> np.ndarray:
    """Performs triangulation for sets of matched points in two images

    Args:
        K (np.ndarray): camera matrix
        Rtw1 (np.ndarray): SE(3) transformation world to camera 1
        Rtw2 (np.ndarray): SE(3) transformation world to camera 2
        points1 (np.ndarray): point coordinates in image 1
        points2 (np.ndarray): point coordinates in image 2

    Returns:
        np.ndarray: triangulated 3D points
    """
    P1 = K @ Rtw1[:3]
    P2 = K @ Rtw2[:3]

    points3D = np.array([DLT(P1, P2, points1[i], points2[i]) for i in range(len(points1))])

    return points3D


def get_homography_pose(Rt1w: np.ndarray, Rt2w: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Calculate homography from 2 relative image poses. Used as ground-truth.
    Homography is calculated for plane at (0, 0, 0) with normal (0, 0, 1)

    Args:
        Rt1w (np.ndarray): SE(3) transform image 1 to world
        Rt2w (np.ndarray): SE(3) transform image 2 to world
        K (np.ndarray): Camera matrix

    Returns:
        np.ndarray: homography matrix
    """
    Rtw1 = np.linalg.inv(Rt1w)
    Rtw2 = np.linalg.inv(Rt2w)

    # Calculate relative pose, transforms points from C2 to C1
    Rt21 = Rtw1 @ Rt2w
    t21 = Rt21[:3, 3]
    R21 = Rt21[:3, :3]

    normal = Rtw2[:3, :3] @ np.array([0, 0, 1])
    normal = normal / np.linalg.norm(normal)

    origin = Rtw2 @ np.array([0, 0, 0, 1])
    origin = origin[:3] / origin[3]

    distance = abs(normal @ origin)

    H = K @ (R21 - t21.reshape(3, 1) @ normal.reshape(1, 3) / distance) @ np.linalg.inv(K)

    H = H / H[2][2]

    return H


def add_noise_to_se3(se3_matrix: np.ndarray, translation_std: float = 0.01, rotation_std: float = 0.001) -> np.ndarray:
    """
    Adds translation and rotation noise to a 4x4 SE(3) transformation matrix. Simulates noise from visual odometry

    Parameters:
    - se3_matrix (numpy.ndarray): SE(3) matrix to add noise.
    - translation_std (float): standard deviation of the Gaussian noise added to translation.
    - rotation_std (float): standard deviation (in radians) of the noise added to rotation angles.

    Returns:
    - numpy.ndarray: 4x4 noisy SE(3) matrix.
    """
    # Extract the original rotation and translation
    original_rotation = se3_matrix[:3, :3]
    original_translation = se3_matrix[:3, 3]

    # Add translation noise
    translation_noise = np.random.normal(0, translation_std, 3)
    noisy_translation = original_translation + translation_noise

    # Add rotation noise
    rotation_noise_vector = np.random.normal(0, rotation_std, 3)
    rotation_noise = R.from_rotvec(rotation_noise_vector).as_matrix()  # type: ignore
    noisy_rotation = rotation_noise @ original_rotation

    # Construct the noisy SE(3) matrix
    noisy_se3 = np.eye(4)
    noisy_se3[:3, :3] = noisy_rotation
    noisy_se3[:3, 3] = noisy_translation

    return noisy_se3


def get_homography_matching(
    image1: MatLike,
    image2: MatLike,
    Rt1w: np.ndarray,
    Rt2w: np.ndarray,
    K: np.ndarray,
    epipolar_threshold: Optional[float] = None,
    triangulation_theshold: Optional[float] = None,
) -> np.ndarray:
    """Calculate homography by matching ORB features and optionaly filtering them
    via epipolar filtering and triangulation thresholding

    Args:
        image1 (MatLike): image 1
        image2 (MatLike): image 2
        Rt1w (np.ndarray): SE(3) transform image 1 to world
        Rt2w (np.ndarray): SE(3) transform image 2 to world
        K (np.ndarray): camera matrix
        epipolar_threshold (Optional[float], optional): max distance to epipolar line in pixels. Defaults to None.
        triangulation_theshold (Optional[float], optional): max distance z distance to plane. Defaults to None.

    Returns:
        np.ndarray: estimated homography
    """
    # Add noise to SE(3) transforms, simulating visual odometry

    Rt12 = np.linalg.inv(Rt2w) @ Rt1w
    t12 = Rt12[:3, 3]
    R12 = Rt12[:3, :3]
    # Essential matrix
    E = skew(t12 / np.linalg.norm(t12)) @ R12
    # Fundamental matrix
    F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)

    # Match ORB featues via descriptor matching
    keypoints1, keypoints2, matches = detect_match_features(image1, image2, n_features=3000, n_matches=None)

    # Convert keypoints to np arrays
    points1, points2 = matches_to_points(keypoints1, keypoints2, matches)

    # Filter matches via epipolar constraints
    if epipolar_threshold is not None:
        points1, points2, _ = filter_matches(points1, points2, matches, F, epipolar_threshold)

    # Threshold matches according to maximum height given by triangulation threshold
    if triangulation_theshold is not None:
        triangulated_points = triangulation(K, np.linalg.inv(Rt1w), np.linalg.inv(Rt2w), points1, points2)
        points1 = points1[np.abs(triangulated_points[:, 2]) < triangulation_theshold]
        points2 = points2[np.abs(triangulated_points[:, 2]) < triangulation_theshold]

    H, _ = cv2.findHomography(points2, points1, cv2.RANSAC)

    return H


def reconstruction_error(H_gt: np.ndarray, H: np.ndarray, img_shape: Tuple[int, int]) -> float:
    """Calculates reconstruction error in pixels, comparing ground-truth and estimated homography.

    Args:
        H_gt (np.ndarray): ground-truth homography
        H (np.ndarray): estimated homography
        img_shape (Tuple[int, int]): image dimensions

    Returns:
        float: MSE in pixels
    """
    xv, yv = np.meshgrid(range(img_shape[1]), range(img_shape[0]), indexing="ij")
    coords = np.vstack([xv.flatten(), yv.flatten(), np.ones(img_shape[0] * img_shape[1])])

    warp_gt = np.dot(H_gt, coords)
    warp_gt /= warp_gt[2, :]

    warp = np.dot(H, coords)
    warp /= warp[2, :]

    mask = np.logical_and(warp_gt[0, :] < img_shape[1], warp_gt[1, :] < img_shape[0])

    warp_gt = warp_gt[:2, mask]
    warp = warp[:2, mask]

    l2 = np.sqrt((warp[0] - warp_gt[0]) ** 2 + (warp[1] - warp_gt[1]) ** 2)

    return np.mean(l2)


def draw_points(
    img: MatLike, points: np.ndarray, point_size: int = 20, color: Tuple[int, int, int] = (0, 0, 255)
) -> MatLike:
    """Draws points on image.

    Args:
        img (MatLike): image
        points (np.ndarray): points coordinates
        point_size (int, optional): point size. Defaults to 20.
        color (Tuple[int, int, int], optional): point color. Defaults to (0, 0, 255).

    Returns:
        MatLike: image with points
    """
    img_pt = img.copy()
    for point in points:
        img_pt = cv2.circle(img_pt, tuple(point), point_size, color, -1)
    return img_pt


def distance_from_line(point: np.ndarray, line: np.ndarray) -> float:
    """Distance from point to line

    Args:
        point (np.ndarray): point in image coordinates
        line (np.ndarray): line in image coordinates

    Returns:
        float: distance
    """
    return abs(point[0] * line[0] + point[1] * line[1] + line[2]) / math.sqrt(line[0] ** 2 + line[1] ** 2)


def skew(vector: np.ndarray) -> np.ndarray:
    """Calculates skew symmetric matrix

    Args:
        vector (np.ndarray): 3D vector

    Returns:
        np.ndarray: Skew-symmetric matrix
    """
    return np.array([[0, -vector[2], vector[1]], [vector[2], 0, -vector[0]], [-vector[1], vector[0], 0]])


def camera_pose_from_target(target: np.ndarray, camera_position: np.ndarray) -> np.ndarray:
    """Calculates homogenous SE(3) transformation for camera at camera_position pointing at target

    Args:
        target (np.ndarray): 3D position of target in world coordinates
        camera_position (np.ndarray): 3D position of camera in world coordinates

    Returns:
        np.ndarray: SE(3) matrix
    """
    z_axis = (target - camera_position) / np.linalg.norm(target - camera_position)
    x_axis = np.cross(np.array([0, 0, -1]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
    homogenous_matrix = np.hstack([rotation_matrix, camera_position.reshape(3, 1)])
    return np.vstack((homogenous_matrix, np.array([0, 0, 0, 1])))


def create_camera_matrix(focal_length_mm: float, sensor_size_mm: float, image_dimensions: np.ndarray) -> np.ndarray:
    """Creates camera matrix from camera parameters

    Args:
        focal_length_mm (float): focal length in mm
        sensor_size_mm (float): senzor size in mm
        image_dimensions (np.ndarray): image dimensions in pixels

    Returns:
        np.ndarray: camera matrix
    """
    fx = fy = focal_length_mm * (image_dimensions[0] / sensor_size_mm)
    cx, cy = image_dimensions[0] / 2, image_dimensions[1] / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def matches_to_points(
    keypoints1: np.ndarray, keypoints2: np.ndarray, matches: List[cv2.DMatch]
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert detected features and matches into corresponding image coordinates

    Args:
        keypoints1 (np.ndarray): keypoints in image 1
        keypoints2 (np.ndarray): keypoints in image 2
        matches (List[cv2.DMatch]): list of matches established between keypoints

    Returns:
        Arrays of corresponding image coordinates
    """
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    return np.array(points1), np.array(points2)


def calculate_metrics(num_images: int, window_size: int):
    """Function to calculate reconstruction error for all frames"""
    frame_0 = 1
    image0 = cv2.imread("OneRevolution/frame_{}.png".format(frame_0))

    focal_length_mm = 750
    sensor_size_mm = 36
    image_dimensions = image0.shape[1::-1]

    # Conversion of focal length from mm to pixels
    focal_length_px = (focal_length_mm / sensor_size_mm) * image_dimensions[0]

    # Camera intrinsic matrix
    K = create_camera_matrix(focal_length_px, sensor_size_mm, image_dimensions)

    with open("cam_path.csv", "r") as csvfile:
        data = list(csv.reader(csvfile, delimiter=","))
    t0 = np.array(data[frame_0], dtype=np.float32)
    Rt0w = camera_pose_from_target(np.array([0, 0, 0]), t0)

    errors = []

    # Create homography estimator
    homography_estimator = IncrementalHomography(window_size=window_size, image0=image0, Rt0w=Rt0w)

    for frame in range(2, num_images):
        image = cv2.imread("OneRevolution/frame_{}.png".format(frame))

        t = np.array(data[frame], dtype=np.float32)

        Rtcw = camera_pose_from_target(np.array([0, 0, 0]), t)

        # Calculate ground truth homography from poses
        H_gt = get_homography_pose(Rt0w, Rtcw, K)

        # Calculate homography from feature matching
        H = homography_estimator.calculate_incremental_homography(
            image=image, Rtcw=Rtcw, K=K, filtered=True, triangulation_threshold=0.05
        )
        # Calculate errors
        errors.append(reconstruction_error(H_gt, H, image0.shape))

    plt.figure()
    plt.plot(range(2, num_images), errors, label="Window size = {}".format(window_size))
    plt.legend()
    plt.ylabel("Reconstruction error in pixels")
    plt.xlabel("Frame #")
    plt.savefig("results/metrics.png")


def create_video(num_images: int, window_size: int):
    """Function to create a video"""
    frame_0 = 1
    image0 = cv2.imread("OneRevolution/frame_{}.png".format(frame_0))

    focal_length_mm = 750
    sensor_size_mm = 36
    image_dimensions = image0.shape[1::-1]

    # Conversion of focal length from mm to pixels
    focal_length_px = (focal_length_mm / sensor_size_mm) * image_dimensions[0]

    # Camera intrinsic matrix
    K = create_camera_matrix(focal_length_px, sensor_size_mm, image_dimensions)

    with open("cam_path.csv", "r") as csvfile:
        data = list(csv.reader(csvfile, delimiter=","))

    t0 = np.array(data[frame_0], dtype=np.float32)
    Rt0w = camera_pose_from_target(np.array([0, 0, 0]), t0)

    homography_estimator = IncrementalHomography(window_size=window_size, image0=image0, Rt0w=Rt0w)

    video_H, video_W = 300, 300
    gap = 50
    video = cv2.VideoWriter(
        "results/Incremental.avi", cv2.VideoWriter_fourcc(*"MJPG"), 20, (video_W * 3 + 2 * gap, video_H)
    )
    video_img = np.zeros((video_H, video_W * 3 + 2 * gap, 3), dtype=np.uint8)

    for frame in range(2, num_images):
        image = cv2.imread("OneRevolution/frame_{}.png".format(frame))

        t = np.array(data[frame], dtype=np.float32)

        Rtcw = camera_pose_from_target(np.array([0, 0, 0]), t)

        H = homography_estimator.calculate_incremental_homography(
            image=image, Rtcw=Rtcw, K=K, filtered=True, triangulation_threshold=0.05
        )
        warped_image = cv2.warpPerspective(image, H, (image0.shape[1], image0.shape[0]))

        image1_resized = cv2.resize(image0, (video_H, video_W))
        image2_resized = cv2.resize(image, (video_H, video_W))
        warped_image_resized = cv2.resize(warped_image, (video_H, video_W))

        video_img[:, :video_W, :] = image1_resized
        video_img[:, video_W + gap : 2 * video_W + gap, :] = warped_image_resized
        video_img[:, 2 * video_W + 2 * gap : 3 * video_W + 2 * gap, :] = image2_resized

        video.write(video_img)

    cv2.destroyAllWindows()
    video.release()


def main():
    # Create video with aligned images
    create_video(num_images=30, window_size=20)

    # Calculate and plot reconstruction error. Note, it is kinda slow
    calculate_metrics(num_images=20, window_size=20)


if __name__ == "__main__":
    main()
