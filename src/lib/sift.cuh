#pragma once
#include <vector>

class Sift {
public:
    Sift(int num_levels, int kernel_size, const std::vector<float>& sigmas);
    ~Sift();
    std::vector<std::vector<int>> detectKeypoints(const unsigned char* img_in, int width, int height, int channels);

private:
    int num_levels_;
    int kernel_size_;
    std::vector<float> sigmas_;

    void gaussianBlur(const unsigned char* img_in, unsigned char* img_out, int width, int height, int channels, const float* kernel, int kernel_size);
    void createGaussianKernel(float* kernel, int kernel_size, float sigma);
    void differenceOfGaussians(const unsigned char* img1, const unsigned char* img2, float* dog, int width, int height, int channels);
    void gaussianPyramidAndDoG(const unsigned char* img_in, int width, int height, int channels, int num_levels, const float* sigmas, unsigned char* blurred, float* dogs, int kernel_size);
    int findDoGKeypoints(const float* dogs, int width, int height, int channels, int num_levels, int (*keypoints)[4], int max_keypoints, float threshold);
};
