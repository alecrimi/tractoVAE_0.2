import SimpleITK as sitk
from pathlib import Path

# Define input/output directories
input_dir = "in_plane"
output_file = "registered_stack.tif"
Path(output_file).parent.mkdir(parents=True, exist_ok=True)

# Load all TIFF files into a list
tiff_files = sorted([str(f) for f in Path(input_dir).glob("*.tif")])
images = [sitk.ReadImage(f) for f in tiff_files]

# Convert the first image to float32 (as the initial reference image)
reference_image = sitk.Cast(images[0], sitk.sitkFloat32)

# Initialize registration parameters
registration = sitk.ImageRegistrationMethod()
registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
registration.SetInterpolator(sitk.sitkLinear)

# Use a rigid 2D transformation
registration.SetInitialTransform(sitk.CenteredTransformInitializer(
    reference_image,
    reference_image,
    sitk.Euler2DTransform(),  # Use 2D transform for 2D images
    sitk.CenteredTransformInitializerFilter.GEOMETRY
))
registration.SetOptimizerScalesFromPhysicalShift()

# List to store aligned images
aligned_images = [reference_image]  # Start with the reference image

# Process each image in the stack (starting from the second image)
for i in range(1, len(images)):
    print(f"Registering image {i+1}/{len(images)}...")

    # Convert the moving image to float32
    moving_image = sitk.Cast(images[i], sitk.sitkFloat32)

    # Perform registration to the previous image (previously aligned)
    transform = registration.Execute(reference_image, moving_image)
    aligned_image = sitk.Resample(moving_image, reference_image, transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

    # Append the aligned image to the list
    aligned_images.append(aligned_image)

    # Update the reference image to the newly aligned image
    reference_image = aligned_image

# Combine all aligned images into a multi-page TIFF
print("Saving the registered stack as a single TIFF file...")
multi_page_image = sitk.JoinSeries(aligned_images)
sitk.WriteImage(multi_page_image, output_file, useCompression=True)

print(f"Registered image stack saved to {output_file}.")
