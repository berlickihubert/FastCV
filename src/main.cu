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
#include "getImgKeypoints.cu"


using namespace cimg_library;

void generateKeypointImage(const std::string& img_path, const std::vector<SiftKeypoint>& keypoints) {
    unsigned char octave_colors[3][3] = {
        {255, 0, 0},
        {0, 255, 0},
        {0, 0, 255}
    };

    CImg<unsigned char> img(img_path.c_str());
    CImg<unsigned char> img_marked = img.get_normalize(0,255);

    for (const auto& kp : keypoints) {
        float scale = std::pow(2.0f, kp.octave);
        int x = int(kp.x * scale);
        int y = int(kp.y * scale);
        unsigned char* color = octave_colors[kp.octave % 3];
        img_marked.draw_circle(x, y, 5, color, 1.0f);
        int len = 20;
        float angle_rad = kp.orientation * 3.14159265f / 180.0f;
        int x2 = int(x + len * std::cos(angle_rad));
        int y2 = int(y + len * std::sin(angle_rad));
        img_marked.draw_line(x, y, x2, y2, color, 1.0f);
    }
    
    std::string output_path = img_path;
    size_t dot_pos = output_path.find_last_of('.');
    output_path.insert(dot_pos, "_keypoints");
    img_marked.save(output_path.c_str());
    std::cout << "Saved: " << output_path << " with " << keypoints.size() << " keypoints (all octaves)." << std::endl;
}

int main() {
    std::vector<std::string> img_paths = {
        "resources/test_images/cat_photo_2_perspective_1.jpg",
        "resources/test_images/cat_photo_2_perspective_2.jpg"
    };

    unsigned char octave_colors[3][3] = {
        {255, 0, 0},
        {0, 255, 0},
        {0, 0, 255}
    };

    for (const auto& img_path : img_paths) {
        auto keypoints = getImgKeypoints(img_path);
        generateKeypointImage(img_path, keypoints);
    }
    return 0;
}
