#define cimg_use_jpeg
#define cimg_use_png
#define cimg_use_tiff
#define cimg_use_gif
#include <iostream>
#include <vector>
#include "lib/matrix_mul.cuh"
#include "lib/sift.cuh"
#include "external/CImg.h"
#include "utils/utils.h"


using namespace cimg_library;

int main() {
    CImg<unsigned char> img("resources/test_images/cat_photo_1.jpg");
    int w = img.width(), h = img.height();
    // Convert to grayscale
    CImg<unsigned char> img_gray = img.get_RGBtoYCbCr().get_channel(0);
    std::vector<unsigned char> img_in(w * h);
    cimg_forXY(img_gray, x, y) {
        img_in[y * w + x] = img_gray(x, y);
    }

    // Gaussian pyramid and DoG (single channel)
    const int num_levels = 6;
    const int kernel_size = 13;
    float sigmas[num_levels] = {1.6f, 2.0f, 2.8f, 4.0f, 5.6f, 8.0f};
    std::vector<unsigned char> blurred(w * h * num_levels);
    std::vector<float> dogs(w * h * (num_levels - 1));
    gaussianPyramidAndDoG(img_in.data(), w, h, 1, num_levels, sigmas, blurred.data(), dogs.data(), kernel_size);

    // Find keypoints
    const int max_keypoints = 10000;
    int keypoints[max_keypoints][4];
    int n_keypoints = findDoGKeypoints(dogs.data(), w, h, 1, num_levels, keypoints, max_keypoints, 10.0f);

    // Mark keypoints on grayscale image (red dots)
    CImg<unsigned char> img_marked = img_gray.get_normalize(0,255);
    for (int i = 0; i < n_keypoints; ++i) {
        int x = keypoints[i][0];
        int y = keypoints[i][1];
        unsigned char color[3] = {255, 0, 0};
        img_marked.draw_circle(x, y, 2, color, 1.0f);
    }
    img_marked.save("resources/test_images/cat_photo_1_keypoints.jpg");
    std::cout << "Saved: resources/test_images/cat_photo_1_keypoints.jpg with " << n_keypoints << " keypoints." << std::endl;
    return 0;
}
