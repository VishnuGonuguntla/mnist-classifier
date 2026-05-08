#include <Eigen/Dense>
#include <fstream>
#include <iostream>

#include "DataReader.h"

int main(int argc, const char *argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <source_file> <output_file> <index>" << std::endl;
        return 1;
    }
    std::string source = argv[1], output = argv[2];
    int index = atoi(argv[3]);

    DataReader parser;
    // MatrixSingle pixel_data = MatrixSingle::Zero(60000, 784);
    MatrixSingle out_file = MatrixSingle::Zero(787, 1);
    MatrixSingle *parsed_image = parser.parse_image(source);
    std::ofstream file(output);
    if (!file.is_open()) {
        std::cout << "Error opening file" << std::endl;
        return 1;
    }

    file << 2 << std::endl;
    file << 28 << std::endl;
    file << 28 << std::endl;
    for (int i = 0; i < 784; i++) {
        file << parsed_image->operator()(index, i) << std::endl;
    }
    file.close();
    return 0;
}