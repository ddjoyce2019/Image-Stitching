import cv2
import numpy as np
import matplotlib.pyplot as plt
#import imutils

np.random.seed(0)

def get_correspondence(img1, img2):
    """
    Args
    img1: left image
    img2: right image

    Output
    points1: coordinates of matched keypoints in img1 | Shape (N, 2), where N is the number of keypoints detected in img1, having corresponding matched keypoint in img2
    points2: coordinates of matched keypoints in img2 | Shape (N, 2)
    """
    ## TODO: Complete this function
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    sift = cv2.SIFT.create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches])

    return points1, points2
    



def visualize_keypoints_and_correspondences(img1, img2, points1, points2):
    """
    Args
    img1: left image
    img2: right image
    points1: coordinates of matched keypoints in img1 | Shape (N, 2), where N is the number of keypoints detected in img1, having corresponding matched keypoint in img2
    points2: coordinates of matched keypoints in img2 | Shape (N, 2)
    """
    # Convert the coordinates to integers
    points1_int = np.int32(points1)
    points2_int = np.int32(points2)

    # Find the size of the output canvas
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    canvas_height = max(h1, h2)
    canvas_width = w1 + w2

    # Create a blank canvas
    img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place the images on the canvas
    img[:h1, :w1, :] = img1
    img[:h2, w1:, :] = img2

    # Draw correspondences
    for i in range(len(points1_int)):
        point1 = tuple(points1_int[i])
        point2 = tuple(points2_int[i] + np.array([w1, 0]))  # Add width of img1 to x-coordinate of points2
        img = cv2.line(img, point1, point2, (0, 255, 0), 1)

    plt.figure(figsize=(20, 10))
    plt.imshow(img)
    plt.title("Correspondences")
    plt.show()


def get_homography(points1, points2):
    """
    Args
    points1: coordinates of keypoints in img1 | Shape (N, 2), where N is the number of keypoints detected in img1, which has corresponding matched keypoint in img2
    points2: coordinates of keypoints in img2 | Shape (N, 2)

    Output
    H: homography matrix | Shape (3, 3)

    NOTE - Hint:
    You can use a RANSAC-based method to robustly estimate the homography matrix H.
    """
    ## TODO: Complete this function

    inlier_T = 5

    best_H = None
    max_inliers = 0

    for _ in range(1000):
        # Randomly select 4 points
        indices = np.random.choice(len(points1), 4, replace=False)
        src_sample = np.array([points1[i] for i in indices])
        dst_sample = np.array([points2[i] for i in indices])

        # Find H
        A = []
        for i in range(4):
            x, y = src_sample[i][0], src_sample[i][1]
            u, v = dst_sample[i][0], dst_sample[i][1]
            A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
            A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])

        A = np.array(A)
        _, _, V = np.linalg.svd(A)
        H = V[-1].reshape(3, 3)
        H /= H[2, 2]

        # Apply H to all points1
        transformed_src = np.array([np.dot(H, np.append(src_point, 1)) for src_point in points1])
        transformed_src /= transformed_src[:, 2][:, None]  # Normalize

        # Find Euclidean Distances
        distances = np.linalg.norm(transformed_src[:, :2] - points2, axis=1)

        # Count inliers based on the inlier_T
        inliers = np.sum(distances < inlier_T)

        # Update H if better one found
        if inliers > max_inliers:
            max_inliers = inliers
            best_H = H


    print(best_H)
    return best_H
        


def stitch(img1, img2, H):
    """
    Args
    img1: left image
    img2: right image
    H: homography matrix

    Output
    img: stitched image

    NOTE - Hint: 
    The homography matrix H computed from get_homography() does not account for translation needed to map the entire output into a single canvas. 
    Hence, take the min and max coordinate ranges (or dimensions) of left and right images to estimate the bounds (min - (x_min, y_min) and max coordinates - (x_max, y_max)) for the final canvas image.
    You might need to incorporate this translation of (x_min, y_min) for warping the final image to the canvas.
    """
    ## TODO: Complete this function
    result_width = img1.shape[1] + img2.shape[1]
    result_height = img1.shape[0]
    
    result = cv2.warpPerspective(img1, H, (result_width, result_height))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2
    
    return result
	



if __name__ == "__main__":
    image1 = cv2.imread("left.jpg")
    image2 = cv2.imread("right.jpg")

    # Rotation Analysis Code
    #cv2.imshow('Original Image', image2)
    height, width = image2.shape[:2]

    # Define the rotation angle (in degrees)
    # angle = 360
    # rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    # image2 = cv2.warpAffine(image2, rotation_matrix, (width, height))
    #cv2.imshow('Rotated Image', image2)

    # Histogram Code
    # angles = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
    # correspondences = [441, 383, 298, 304, 401, 425, 429, 386, 309, 307, 399, 426]

    # plt.xlabel("Angles")
    # plt.ylabel("Number of Correspondences")
    # plt.title("Number of Correspondences vs. Rotation Angle")
    # plt.bar(angles, correspondences)


    # Scale Analysis Code
    # scale_factor = 0.5
    # new_width = int(image2.shape[1] * scale_factor)
    # new_height = int(image2.shape[0] * scale_factor)
    # image2 = cv2.resize(image2,(new_width,new_height))

    # Illumination Analysis Code
    # cv2.imshow('Original Image', image2)
    # brightness_increase = -150  # Change this value to increase/decrease brightness
    # image2 = np.clip(image2.astype(np.int32) + brightness_increase, 0, 255).astype(np.uint8)

    # cv2.imshow('Brightened Image', image2)

    points1, points2 = get_correspondence(image1, image2)
    print(f"{len(points1)}, {len(points2)}")
    visualize_keypoints_and_correspondences(image1, image2, points1, points2)
    
    H = get_homography(points2, points1)
    output = stitch(image2, image1, H)
    cv2.imwrite("output.jpg", output)