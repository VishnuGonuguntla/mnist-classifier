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
    // MatrixInteger label_data(60000, 10);

    MatrixInteger *parsed_label = parser.parse_label(source);
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

// #include <Eigen/Dense>
// #include <fstream>
// #include <iostream>
// #include "DataReader.h"

// int main(int argc, const char* argv[]) {
//     if (argc < 4) {
//         std::cerr << "Usage: " << argv[0] << " <source> <output> <index>" << std::endl;
//         return 1;
//     }

//     std::string source = argv[1];
//     std::string output = argv[2];
//     int index          = atoi(argv[3]);

//     DataReader parser;
//     MatrixInteger* parsed_label = parser.parse_label(source);

//     if (parsed_label == nullptr) {
//         std::cerr << "Failed to parse label file: " << source << std::endl;
//         return 1;
//     }
//     std::cout << parsed_label->rows() << std::endl;

//     if (index < 0 || index >= parsed_label->rows()) {
//         std::cerr << "Index " << index << " out of range (max: "
//                   << parsed_label->rows() - 1 << ")" << std::endl;
//         return 1;
//     }

//     std::ofstream file(output);
//     if (!file.is_open()) {
//         std::cerr << "Error opening output file: " << output << std::endl;
//         return 1;
//     }

//     file << 10 << std::endl;   // number of classes
//     for (int i = 0; i < 10; i++) {
//         file << (*parsed_label)(index, i) << std::endl;
//     }

//     file.close();
//     return 0;
// }