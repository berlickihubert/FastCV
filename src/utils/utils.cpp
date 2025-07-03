#include <iostream>
#include "utils.h"

void print_matrix(const float* mat, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << mat[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}
