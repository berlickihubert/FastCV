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

SiftOctave::SiftOctave(int num_levels, int kernel_size, const std::vector<float>& sigmas)
    : num_levels_(num_levels), kernel_size_(kernel_size), sigmas_(sigmas) {}

SiftOctave::~SiftOctave() {}


void SiftOctave::createGaussianKernel(float* kernel, int kernel_size, float sigma) {
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

void SiftOctave::gaussianBlur(const unsigned char* img_in, unsigned char* img_out, int width, int height, int channels, const float* kernel, int kernel_size) {
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
void SiftOctave::differenceOfGaussians(const unsigned char* img1, const unsigned char* img2, float* dog, int width, int height, int channels) {
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
void SiftOctave::gaussianPyramidAndDoG(const unsigned char* img_in, int width, int height, int channels, int num_levels, const float* sigmas, unsigned char* blurred, float* dogs, int kernel_size) {
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
void SiftOctave::findDoGKeypoints(const float* dogs, int width, int height, int channels, int num_levels, std::vector<SiftKeypoint>& keypoints, int max_keypoints, float threshold) {
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

                        dog_curr[((y - 1) * width + (x - 1)) * channels + c],
                        dog_curr[((y - 1) * width + x) * channels + c],
                        dog_curr[((y - 1) * width + (x + 1)) * channels + c],
                        dog_curr[(y * width + (x - 1)) * channels + c],
                        dog_curr[(y * width + (x + 1)) * channels + c],
                        dog_curr[((y + 1) * width + (x - 1)) * channels + c],
                        dog_curr[((y + 1) * width + x) * channels + c],
                        dog_curr[((y + 1) * width + (x + 1)) * channels + c]
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
                        kp.octave = 0; // Placeholder
                        kp.orientation = 0.0f;
                        kp.descriptor.fill(0.0f);
                        keypoints.push_back(kp);
                    }
                }
            }
        }
    }
}


float SiftOctave::assignOrientation(const unsigned char* blurred, int width, int height, int channels, int x, int y, int scale) {
    const int radius = 8;
    const int hist_bins = 36;
    std::vector<float> hist(hist_bins, 0.0f);
    float sigma = 1.5f * sigmas_[scale];
    float two_sigma2 = 2.0f * sigma * sigma;
    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int xx = x + dx;
            int yy = y + dy;
            if (xx <= 0 || xx >= width-1 || yy <= 0 || yy >= height-1) continue;
            float dx_val = float(blurred[yy * width + (xx+1)]) - float(blurred[yy * width + (xx-1)]);
            float dy_val = float(blurred[(yy+1) * width + xx]) - float(blurred[(yy-1) * width + xx]);
            float mag = std::sqrt(dx_val * dx_val + dy_val * dy_val);
            float angle = std::atan2(dy_val, dx_val) * 180.0f / 3.14159265f;
            if (angle < 0) angle += 360.0f;
            float weight = std::exp(-(dx*dx + dy*dy) / two_sigma2);
            int bin = int(std::round(angle / 360.0f * hist_bins)) % hist_bins;
            hist[bin] += mag * weight;
        }
    }
    int max_bin = std::max_element(hist.begin(), hist.end()) - hist.begin();
    float orientation = max_bin * 360.0f / hist_bins;
    return orientation;
}


std::array<float, 128> SiftOctave::computeDescriptor(const unsigned char* blurred, int width, int height, int channels, int x, int y, int scale, float orientation) {
    std::array<float, 128> desc = {0};
    const int d = 4;
    const int n = 8;
    const int win_size = 16;
    float cos_t = std::cos(orientation * 3.14159265f / 180.0f);
    float sin_t = std::sin(orientation * 3.14159265f / 180.0f);
    float sigma = 0.5f * win_size;
    float two_sigma2 = 2.0f * sigma * sigma;
    for (int i = -win_size/2; i < win_size/2; ++i) {
        for (int j = -win_size/2; j < win_size/2; ++j) {
            float rx =  cos_t * j + sin_t * i;
            float ry = -sin_t * j + cos_t * i;
            int bin_x = int((rx + win_size/2 - 0.5f) / (win_size/d));
            int bin_y = int((ry + win_size/2 - 0.5f) / (win_size/d));
            if (bin_x < 0 || bin_x >= d || bin_y < 0 || bin_y >= d) continue;
            int xx = x + j;
            int yy = y + i;
            if (xx <= 0 || xx >= width-1 || yy <= 0 || yy >= height-1) continue;
            float dx_val = float(blurred[yy * width + (xx+1)]) - float(blurred[yy * width + (xx-1)]);
            float dy_val = float(blurred[(yy+1) * width + xx]) - float(blurred[(yy-1) * width + xx]);
            float mag = std::sqrt(dx_val * dx_val + dy_val * dy_val);
            float angle = std::atan2(dy_val, dx_val) * 180.0f / 3.14159265f - orientation;
            while (angle < 0) angle += 360.0f;
            while (angle >= 360.0f) angle -= 360.0f;
            float weight = std::exp(-(rx*rx + ry*ry) / two_sigma2);
            int bin_o = int(std::round(angle / 360.0f * n)) % n;
            int idx = (bin_y * d + bin_x) * n + bin_o;
            if (idx >= 0 && idx < 128)
                desc[idx] += mag * weight;
        }
    }
    float norm = 0.0f;
    for (float v : desc) norm += v*v;
    norm = std::sqrt(norm);
    if (norm > 1e-6f) for (float& v : desc) v /= norm;
    for (float& v : desc) if (v > 0.2f) v = 0.2f;
    norm = 0.0f;
    for (float v : desc) norm += v*v;
    norm = std::sqrt(norm);
    if (norm > 1e-6f) for (float& v : desc) v /= norm;
    return desc;
}


std::vector<SiftKeypoint> SiftOctave::detectKeypoints(const unsigned char* img_in, int width, int height, int channels) {
    std::vector<SiftKeypoint> keypoints;
    int img_size = width * height * channels;
    std::vector<unsigned char> blurred(img_size * num_levels_);
    std::vector<float> dogs(img_size * (num_levels_ - 1));
    gaussianPyramidAndDoG(img_in, width, height, channels, num_levels_, sigmas_.data(), blurred.data(), dogs.data(), kernel_size_);

    const int max_keypoints = 10000;
    findDoGKeypoints(dogs.data(), width, height, channels, num_levels_, keypoints, max_keypoints, 10.0f);

    for (auto& kp : keypoints) {
        kp.orientation = assignOrientation(blurred.data() + kp.scale * width * height, width, height, channels, kp.x, kp.y, kp.scale);
    }
    return keypoints;
}