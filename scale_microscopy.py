import numpy as np
from skimage import io, transform
import tifffile as tiff

# --- PARAMETERS ---
input_path = 'microscopy_stack.tif'       # Input TIFF stack
output_path = 'microscopy_stack_resized.tif'  # Output resized TIFF
scale = 0.5                               # Uniform scale factor (e.g., 0.5 = half size)


print("Loading image stack...")
image_stack = io.imread(input_path)  # shape: (Z, Y, X)
print(f"Original shape: {image_stack.shape}")

if isinstance(scale, (float, int)):
    scale_factors = (scale, scale, scale)
else:
    scale_factors = scale

# skimage.transform.rescale expects (Z, Y, X) scaling
print(f"Scaling factors: {scale_factors}")
print("Resizing (this may take some time)...")
resized_stack = transform.rescale(
    image_stack,
    scale_factors,
    order=1,              # 1 = bilinear (good tradeoff between speed/quality)
    preserve_range=True,  # Keep intensity range the same
    anti_aliasing=True,
    channel_axis=None     # Important: treat as grayscale stack
).astype(image_stack.dtype)

print(f"Resized shape: {resized_stack.shape}")

print("Saving resized stack...")
tiff.imwrite(output_path, resized_stack)
print(f"Saved resized TIFF to: {output_path}")
