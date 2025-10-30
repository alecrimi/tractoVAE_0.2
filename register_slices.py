
#First stack all files together with ImageMagik:
# convert image1.tiff image2.tiff image3.tiff stacked_output.tiff
#Then run this:

import cv2
import numpy as np
from tifffile import imread, imwrite

# Load slides from a multi-page TIFF file
file_path = "stacked_output.tiff"
slides = imread(file_path)  # Load as a stack (NumPy array)

# Select a reference slide (e.g., the first one)
reference = slides[0]

# Initialize a list to store registered slides
registered_slides = []

for i, slide in enumerate(slides):
    if i == 0:
        # First slide is the reference, no transformation needed
        registered_slides.append(reference)
        continue

    # Convert slides to grayscale (if not already)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if len(reference.shape) == 3 else reference
    slide_gray = cv2.cvtColor(slide, cv2.COLOR_BGR2GRAY) if len(slide.shape) == 3 else slide

    # Detect and compute features using ORB
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(ref_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(slide_gray, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    src_pts = np.float32([keypoints2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate transformation matrix (affine or perspective)
    matrix, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # Apply the transformation
    rows, cols = ref_gray.shape
    aligned_slide = cv2.warpAffine(slide, matrix, (cols, rows))

    registered_slides.append(aligned_slide)

# Save the registered slides as a new TIFF file
output_file = "registered_slides.tif"
imwrite(output_file, np.array(registered_slides))

print(f"Registered slides saved as {output_file}")
