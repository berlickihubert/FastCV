#pragma once
#include <vector>
#include <array>

struct SiftKeypoint {
    int x;
    int y;
    int octave;
    int scale;
    float orientation;
    std::array<float, 128> descriptor;
};


class SiftOctave {
public:
    SiftOctave(int num_levels, int kernel_size, const std::vector<float>& sigmas);
    ~SiftOctave();
    std::vector<SiftKeypoint> detectKeypoints(const unsigned char* img_in, int width, int height, int channels);

private:
    int num_levels_;
    int kernel_size_;
    std::vector<float> sigmas_;

    void gaussianBlur(const unsigned char* img_in, unsigned char* img_out, int width, int height, int channels, const float* kernel, int kernel_size);
    void createGaussianKernel(float* kernel, int kernel_size, float sigma);
    void differenceOfGaussians(const unsigned char* img1, const unsigned char* img2, float* dog, int width, int height, int channels);
    void gaussianPyramidAndDoG(const unsigned char* img_in, int width, int height, int channels, int num_levels, const float* sigmas, unsigned char* blurred, float* dogs, int kernel_size);
    void findDoGKeypoints(const float* dogs, int width, int height, int channels, int num_levels, std::vector<SiftKeypoint>& keypoints, int max_keypoints, float threshold);
    float assignOrientation(const unsigned char* blurred, int width, int height, int channels, int x, int y, int scale);
    std::array<float, 128> computeDescriptor(const unsigned char* blurred, int width, int height, int channels, int x, int y, int scale, float orientation);
};
