#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "DataReader.h"

int main(int argc, const char *argv[]) {
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <source_file> <output_file> <index>" << std::endl;
        return 1;
    }
    std::string source = argv[1];
    std::string output = argv[2];
    int index = atoi(argv[3]);

    DataReader parser;
    // MatrixInteger label_data(60000, 10);

    MatrixSingle *parsed_label = parser.parse_label(source);
    std::ofstream file2(output);
    if (!file2.is_open()) {
        std::cout << "Unable to open file" << std::endl;
        return 1;
    }

    file2 << 1 << std::endl;
    file2 << 10 << std::endl;
    for (int i = 0; i < 10; i++) {
        file2 << (*parsed_label)(index, i) << std::endl;
    }
    file2.close();
    
    return 0;
}