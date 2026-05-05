#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "DataReader.h"

int main(int argc, const char *argv[]) {
    std::string source = argv[1];
    std::string output = argv[2];
    int index = atoi(argv[3]);

    DataReader parser;
    MatrixInteger label_data(60000, 10);

    parser.parse_label(label_data, source);
    std::ofstream file2(output);
    if (file2.is_open()) {
        file2 << 1 << std::endl;
        file2 << 10 << std::endl;
        for (int i = 2; i < 12; i++) {
            file2 << label_data(index, i - 2) << std::endl;
        }
        file2.close();
    } else {
        std::cout << "Unable to open file" << std::endl;
    }
    
    return 0;
}