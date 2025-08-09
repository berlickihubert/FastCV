# Documentation for sift.cu and sift.cuh

### Function: `void gaussianBlur(const unsigned char* img_in, unsigned char* img_out, int width, int height, int channels, const float* kernel, int kernel_size);`

**Description:**
Applies a 2D Gaussian blur to an input image using a specified convolution kernel. The operation is performed on the GPU (CUDA), supporting multi-channel images.

**Parameters:**
- `img_in`: Pointer to the input image data (flattened array, size = `width * height * channels`).
- `img_out`: Pointer to the output image data (same size as `img_in`).
- `width`: Image width (in pixels).
- `height`: Image height (in pixels).
- `channels`: Number of color channels (e.g., 1 for grayscale, 3 for RGB).
- `kernel`: Pointer to the Gaussian kernel (flattened 2D array, size = `kernel_size * kernel_size`).
- `kernel_size`: Size of the Gaussian kernel.

**Input:**
- `img_in` should be a contiguous array of unsigned char values representing the image, ordered as [row-major, channel-interleaved].
- `kernel` should be a normalized Gaussian kernel (sum = 1).

**Output:**
- `img_out` will contain the blurred image, same shape and type as `img_in`.

### Function: `void differenceOfGaussians(const unsigned char* img1, const unsigned char* img2, float* dog, int width, int height, int channels);`

**Description:**
Computes the Difference of Gaussians (DoG) between two blurred images. The result is a float-valued image where each pixel is the difference between the corresponding pixels in `img1` and `img2`.

**Parameters:**
- `img1`: Pointer to the first input image (unsigned char array, size = `width * height * channels`).
- `img2`: Pointer to the second input image (same size and format as `img1`).
- `dog`: Pointer to the output DoG image (float array, size = `width * height * channels`).
- `width`: Image width (in pixels).
- `height`: Image height (in pixels).
- `channels`: Number of color channels (e.g., 1 for grayscale, 3 for RGB).

**Input:**
- `img1` and `img2` should be contiguous arrays of unsigned char values, representing two blurred versions of the same image (typically with different Gaussian sigmas).

**Output:**
- `dog` will contain the pixel-wise difference: `dog[i] = float(img1[i]) - float(img2[i])` for all pixels and channels.


### Function: `void gaussianPyramidAndDoG(const unsigned char* img_in, int width, int height, int channels, int num_levels, const float* sigmas, unsigned char* blurred, float* dogs, int kernel_size);`

**Description:**
Builds a Gaussian pyramid and computes the Difference of Gaussians (DoG) images for a given input image. This is a core step in the SIFT pipeline for multi-scale keypoint detection.

**Parameters:**
- `img_in`: Pointer to the input image data (unsigned char array, size = `width * height * channels`).
- `width`: Image width (in pixels).
- `height`: Image height (in pixels).
- `channels`: Number of color channels (e.g., 1 for grayscale, 3 for RGB).
- `num_levels`: Number of pyramid levels (scales) to generate.
- `sigmas`: Pointer to an array of Gaussian standard deviations (length = `num_levels`).
- `blurred`: Pointer to output array for all blurred images (size = `width * height * channels * num_levels`).
- `dogs`: Pointer to output array for all DoG images (size = `width * height * channels * (num_levels - 1)`).
- `kernel_size`: Size of the Gaussian kernel (must be odd).

**Input:**
- `img_in` is the original image, flattened.
- `sigmas` should contain the desired Gaussian sigmas for each pyramid level.

**Output:**
- `blurred` will contain all blurred images, concatenated by level.
- `dogs` will contain all DoG images, concatenated by level.


### Function: `float assignOrientation(const unsigned char* blurred, int width, int height, int channels, int x, int y, int scale);`

**Description:**
Estimates the dominant orientation of a keypoint by analyzing the gradient directions in a local window around the keypoint. This step provides rotation invariance for SIFT keypoints.

**Parameters:**
- `blurred`: Pointer to the blurred image data (unsigned char array, size = `width * height * channels`).
- `width`: Image width (in pixels).
- `height`: Image height (in pixels).
- `channels`: Number of color channels (should be 1 for grayscale SIFT).
- `x`, `y`: Coordinates of the keypoint.
- `scale`: Pyramid scale index (used to select the appropriate Gaussian sigma).

**Input:**
- `blurred` should be the blurred image at the appropriate scale.

**Output:**
- Returns the dominant orientation (in degrees, [0, 360)) for the keypoint at (x, y, scale).

**Example:**
```cpp
float orientation = assignOrientation(blurred, width, height, 1, x, y, scale);
```