import cv2

# Read the grayscale image
img = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)

# Initialize the BRISK detector
brisk = cv2.BRISK_create()

# Detect keypoints
keypoints = brisk.detect(img, None)

# Draw keypoints on the image
img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))

# Show results
cv2.imshow('BRISK Feature points', img_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()
