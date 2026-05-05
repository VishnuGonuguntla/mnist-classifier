#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "DataReader.h"

int main(int argc, const char *argv[]) {
    std::string source = argv[1], output = argv[2];
    int index = atoi(argv[3]);

    DataReader parser;
    MatrixSingle pixel_data = MatrixSingle::Zero(60000, 784);
    MatrixSingle out_file = MatrixSingle::Zero(787, 1);
    parser.parse_image(pixel_data, source);
    std::ofstream file(output);
    if (file.is_open()) {
        file << 2 << std::endl;
        file << 28 << std::endl;
        file << 28 << std::endl;
        for (int i = 0; i < 784; i++) {
            file << pixel_data(index, i) << std::endl;
        }
        file.close();
    } else {
        std::cout << "Error opening file";
    }
    return 0;
}