#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include "sift.cuh"

__global__ void gaussianBlurKernel(const unsigned char* img_in, unsigned char* img_out, int width, int height, int channels, const float* kernel, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < width && y < height && c < channels) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        for (int neighbour_y = max(y - (kernel_size / 2), 0); neighbour_y <= min(y + (kernel_size / 2), height - 1); neighbour_y++) {
            for (int neighbour_x = max(x - (kernel_size / 2), 0); neighbour_x <= min(x + (kernel_size / 2), width - 1); neighbour_x++) {
                float w = kernel[(neighbour_y - (y - (kernel_size / 2))) * kernel_size + (neighbour_x - (x - kernel_size / 2))];
                sum += w * img_in[(neighbour_y * width + neighbour_x) * channels + c];
                weight_sum += w;
            }
        }
        img_out[(y * width + x) * channels + c] = (unsigned char)(sum / weight_sum);
    }
}

__global__ void dogKernel(const unsigned char* img1, const unsigned char* img2, float* dog, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    if (x < width && y < height && c < channels) {
        int idx = (y * width + x) * channels + c;
        dog[idx] = static_cast<float>(img1[idx]) - static_cast<float>(img2[idx]);
    }
}

Sift::Sift(int num_levels, int kernel_size, const std::vector<float>& sigmas)
    : num_levels_(num_levels), kernel_size_(kernel_size), sigmas_(sigmas) {}

Sift::~Sift() {}


void Sift::createGaussianKernel(float* kernel, int kernel_size, float sigma) {
    int k = kernel_size / 2;
    float sum = 0.0f;
    for (int y = 0; y < kernel_size; ++y) {
        for (int x = 0; x < kernel_size; ++x) {
            float dx = x - k;
            float dy = y - k;
            float value = expf(-(dx * dx + dy * dy) / (2 * sigma * sigma));
            kernel[y * kernel_size + x] = value;
            sum += value;
        }
    }
    for (int i = 0; i < kernel_size * kernel_size; ++i) kernel[i] /= sum;
}

void Sift::gaussianBlur(const unsigned char* img_in, unsigned char* img_out, int width, int height, int channels, const float* kernel, int kernel_size) {
    size_t img_size = width * height * channels * sizeof(unsigned char);
    size_t kernel_bytes = kernel_size * kernel_size * sizeof(float);
    unsigned char *d_in, *d_out;
    float *d_kernel;
    cudaMalloc(&d_in, img_size);
    cudaMalloc(&d_out, img_size);
    cudaMalloc(&d_kernel, kernel_bytes);
    cudaMemcpy(d_in, img_in, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((width+15)/16, (height+15)/16, channels);
    gaussianBlurKernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, width, height, channels, d_kernel, kernel_size);
    cudaDeviceSynchronize();
    cudaMemcpy(img_out, d_out, img_size, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);
}

// in: img1, img2
// out: dog
void Sift::differenceOfGaussians(const unsigned char* img1, const unsigned char* img2, float* dog, int width, int height, int channels) {
    size_t img_size = width * height * channels * sizeof(unsigned char);
    size_t dog_size = width * height * channels * sizeof(float);
    unsigned char *d_img1, *d_img2;
    float *d_dog;
    cudaMalloc(&d_img1, img_size);
    cudaMalloc(&d_img2, img_size);
    cudaMalloc(&d_dog, dog_size);
    cudaMemcpy(d_img1, img1, img_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, img2, img_size, cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((width+15)/16, (height+15)/16, channels);
    dogKernel<<<numBlocks, threadsPerBlock>>>(d_img1, d_img2, d_dog, width, height, channels);
    cudaDeviceSynchronize();
    cudaMemcpy(dog, d_dog, dog_size, cudaMemcpyDeviceToHost);
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_dog);
}

//in: img_in
//out: dogs
void Sift::gaussianPyramidAndDoG(const unsigned char* img_in, int width, int height, int channels, int num_levels, const float* sigmas, unsigned char* blurred, float* dogs, int kernel_size) {
    int img_size = width * height * channels;
    std::vector<float> kernel(kernel_size * kernel_size);
    for (int i = 0; i < num_levels; ++i) {
        createGaussianKernel(kernel.data(), kernel_size, sigmas[i]);
        gaussianBlur(img_in, blurred + i * img_size, width, height, channels, kernel.data(), kernel_size);
    }
    for (int i = 0; i < num_levels - 1; ++i) {
        differenceOfGaussians(blurred + i * img_size, blurred + (i + 1) * img_size, dogs + i * img_size, width, height, channels);
    }
}

// std::vector<float> dogs(img_size * (num_levels_ - 1));
void Sift::findDoGKeypoints(const float* dogs, int width, int height, int channels, int num_levels, std::vector<SiftKeypoint>& keypoints, int max_keypoints, float threshold) {
    int img_size = width * height * channels;
    for (int l = 1; l < num_levels - 2; ++l) {
        const float* dog_prev = dogs + (l - 1) * img_size;
        const float* dog_curr = dogs + l * img_size;
        const float* dog_next = dogs + (l + 1) * img_size;
        
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                for (int c = 0; c < channels; ++c) {
                    int idx = (y * width + x) * channels + c;
                    float val = dog_curr[idx];
                    if (fabs(val) < threshold) continue;
                    bool is_max = true, is_min = true;
                    
                    std::vector<float> neighbours = {
                        dog_prev[((y - 1) * width + (x - 1)) * channels + c],
                        dog_prev[((y - 1) * width + x) * channels + c],
                        dog_prev[((y - 1) * width + (x + 1)) * channels + c],
                        dog_prev[(y * width + (x - 1)) * channels + c],
                        dog_prev[(y * width + x) * channels + c],
                        dog_prev[(y * width + (x + 1)) * channels + c],
                        dog_prev[((y + 1) * width + (x - 1)) * channels + c],
                        dog_prev[((y + 1) * width + x) * channels + c],
                        dog_prev[((y + 1) * width + (x + 1)) * channels + c],

                        dog_next[((y - 1) * width + (x - 1)) * channels + c],
                        dog_next[((y - 1) * width + x) * channels + c],
                        dog_next[((y - 1) * width + (x + 1)) * channels + c],
                        dog_next[(y * width + (x - 1)) * channels + c],
                        dog_next[(y * width + x) * channels + c],
                        dog_next[(y * width + (x + 1)) * channels + c],
                        dog_next[((y + 1) * width + (x - 1)) * channels + c],
                        dog_next[((y + 1) * width + x) * channels + c],
                        dog_next[((y + 1) * width + (x + 1)) * channels + c],

                        dog[((y - 1) * width + (x - 1)) * channels + c],
                        dog[((y - 1) * width + x) * channels + c],
                        dog[((y - 1) * width + (x + 1)) * channels + c],
                        dog[(y * width + (x - 1)) * channels + c],
                        dog[(y * width + (x + 1)) * channels + c],
                        dog[((y + 1) * width + (x - 1)) * channels + c],
                        dog[((y + 1) * width + x) * channels + c],
                        dog[((y + 1) * width + (x + 1)) * channels + c]
                    };

                    for(const auto& nval : neighbours) {
                        if (val <= nval) is_max = false;
                        if (val >= nval) is_min = false;
                    }

                    if ((is_max || is_min) && keypoints.size() < static_cast<size_t>(max_keypoints)) {
                        SiftKeypoint kp;
                        kp.x = x;
                        kp.y = y;
                        kp.scale = l;
                        kp.orientation = 0.0f;
                        kp.descriptor.fill(0.0f);
                        keypoints.push_back(kp);
                    }
                }
            }
        }
    }
}

std::vector<SiftKeypoint> Sift::detectKeypoints(const unsigned char* img_in, int width, int height, int channels) {
    std::vector<SiftKeypoint> keypoints;
    int img_size = width * height * channels;
    std::vector<unsigned char> blurred(img_size * num_levels_);
    std::vector<float> dogs(img_size * (num_levels_ - 1));
    gaussianPyramidAndDoG(img_in, width, height, channels, num_levels_, sigmas_.data(), blurred.data(), dogs.data(), kernel_size_);

    const int max_keypoints = 10000;
    findDoGKeypoints(dogs.data(), width, height, channels, num_levels_, keypoints, max_keypoints, 10.0f);
    return keypoints;
}