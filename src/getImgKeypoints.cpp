#include "lib/sift.cuh"
#include "external/CImg.h"
#include <vector>
#include <string>
#include <cmath>

using namespace cimg_library;

std::vector<SiftKeypoint> getImgKeypoints(const std::string& img_path) {
    CImg<unsigned char> img(img_path.c_str());
    int w = img.width(), h = img.height();
    CImg<unsigned char> img_gray = img.get_RGBtoYCbCr().get_channel(0);
    const int num_levels = 6;
    const int kernel_size = 13;
    std::vector<float> sigmas = {1.6f, 2.0f, 2.8f, 4.0f, 5.6f, 8.0f};
    const int num_octaves = 3;
    std::vector<SiftKeypoint> all_keypoints;
    for (int octave = 0; octave < num_octaves; ++octave) {
        float scale = std::pow(2.0f, octave);
        int w_oct = int(w / scale);
        int h_oct = int(h / scale);
        CImg<unsigned char> img_octave = img_gray.get_resize(w_oct, h_oct, -100, -100, 3);
        std::vector<unsigned char> img_in_oct(w_oct * h_oct);
        cimg_forXY(img_octave, x, y) {
            img_in_oct[y * w_oct + x] = img_octave(x, y);
        }
        std::vector<float> octave_sigmas;
        for (float s : sigmas) octave_sigmas.push_back(s * scale);
        SiftOctave siftOctave(num_levels, kernel_size, octave_sigmas);
        auto keypoints = siftOctave.detectKeypoints(img_in_oct.data(), w_oct, h_oct, 1);
        std::cout << "Octave " << octave << ": Detected " << keypoints.size() << " keypoints." << std::endl;
        for (auto& kp : keypoints) {
            kp.x = int(kp.x * scale);
            kp.y = int(kp.y * scale);
            kp.octave = octave;
            all_keypoints.push_back(kp);
        }
    }
    return all_keypoints;
}
