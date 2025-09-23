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
#include "getImgKeypoints.cpp"


using namespace cimg_library;

int main() {
    std::string img_path = "resources/test_images/cat_photo_1.jpg";
    auto keypoints = getImgKeypoints(img_path);
    int total_keypoints_count = keypoints.size();

    CImg<unsigned char> img(img_path.c_str());
    int w = img.width(), h = img.height();
    CImg<unsigned char> img_marked = img.get_normalize(0,255);

    unsigned char octave_colors[3][3] = {
        {255, 0, 0},
        {0, 255, 0},
        {0, 0, 255}
    };

    for (const auto& kp : keypoints) {
            float scale = std::pow(2.0f, kp.octave);
            int x = int(kp.x * scale);
            int y = int(kp.y * scale);
            unsigned char* color = octave_colors[kp.octave % 3];
            img_marked.draw_circle(x, y, 2, color, 1.0f);
            int len = 10;
            float angle_rad = kp.orientation * 3.14159265f / 180.0f;
            int x2 = int(x + len * std::cos(angle_rad));
            int y2 = int(y + len * std::sin(angle_rad));
            img_marked.draw_line(x, y, x2, y2, color, 1.0f);
    }
    img_marked.save("resources/test_images/cat_photo_1_keypoints.jpg");
    std::cout << "Saved: resources/test_images/cat_photo_1_keypoints.jpg with " << total_keypoints_count << " keypoints (all octaves)." << std::endl;
    return 0;
}
