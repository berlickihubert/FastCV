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
    CImg<unsigned char> img_gray = img.get_RGBtoYCbCr().get_channel(0);
    std::vector<unsigned char> img_in(w * h);
    cimg_forXY(img_gray, x, y) {
        img_in[y * w + x] = img_gray(x, y);
    }

    const int num_levels = 6;
    const int kernel_size = 13;
    std::vector<float> sigmas = {1.6f, 2.0f, 2.8f, 4.0f, 5.6f, 8.0f};
    Sift sift(num_levels, kernel_size, sigmas);
    auto keypoints = sift.detectKeypoints(img_in.data(), w, h, 1);

    CImg<unsigned char> img_marked = img_gray.get_normalize(0,255);
    for (const auto& kp : keypoints) {
        int x = kp.x;
        int y = kp.y;
        unsigned char color[3] = {255, 0, 0};
        img_marked.draw_circle(x, y, 2, color, 1.0f);
        
        int len = 10;
        float angle_rad = kp.orientation * 3.14159265f / 180.0f;
        int x2 = int(x + len * std::cos(angle_rad));
        int y2 = int(y + len * std::sin(angle_rad));
        img_marked.draw_line(x, y, x2, y2, color, 1.0f);
    }
    img_marked.save("resources/test_images/cat_photo_1_keypoints.jpg");
    std::cout << "Saved: resources/test_images/cat_photo_1_keypoints.jpg with " << keypoints.size() << " keypoints." << std::endl;
    return 0;
}
