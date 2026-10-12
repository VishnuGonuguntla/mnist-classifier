#include "DataReader.h"


#include <cstdint>
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <vector>

// #define MatrixSingle Eigen::MatrixXf
// #define MatrixInteger Eigen::MatrixXi

DataReader::DataReader(const std::map<std::string, std::string>& config_data) {
    train_i = config_data.at("rel_path_train_images");
    train_l = config_data.at("rel_path_train_labels");
    test_i = config_data.at("rel_path_test_images");
    test_l = config_data.at("rel_path_test_labels");
}

void DataReader::parse_image(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) throw std::runtime_error("Cannot open image file: " + filename);

    auto start = std::chrono::steady_clock::now();
    std::cout << "Parsing image data from: " << filename << "Time: ";
    
    if (bigEndianToHost(file) != 2051 ) throw std::runtime_error("Bad image magic number.");

    const std::uint32_t number_of_images = bigEndianToHost(file);
    const std::uint32_t number_of_rows = bigEndianToHost(file);
    const std::uint32_t number_of_columns = bigEndianToHost(file);
    const std::size_t pix = number_of_rows * number_of_columns;
    
    std::vector<unsigned char> raw(number_of_images * pix);
    file.read(reinterpret_cast<char*>(raw.data()),raw.size());
    if (!file) throw std::runtime_error("Trunacated image file");

    image.resize(number_of_images, pix);
    for (size_t i = 0; i < number_of_images; i++)
        for (size_t p = 0; p < pix; p++)
                image(i, p) = raw[i * pix + p] * (1.0f / 255.0f);

    Helper::calculate_time(start);
}

void DataReader::parse_label(const std::string& filename) {
    std::ifstream file;
    file.open(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cout << "Label Data Not open" << std::endl;
    }
    auto start = std::chrono::steady_clock::now();
    std::cout << "Parsing label data from: " << filename << "Time: ";
    if (bigEndianToHost(file) != 2049 ) throw std::runtime_error("Bad label magic number.");
    const std::uint32_t number_of_images = bigEndianToHost(file);

    // file.read((char *)&number_of_images, sizeof(number_of_images));
    // const std::uint32_t number_of_images = bigEndianToHost(file);
    std::vector<unsigned char> raw(number_of_images);
    // file.read(reinterpret_cast<char*>(raw.data()),raw.size());

    label = MatrixSingle::Zero(number_of_images, 10);
    file.read(reinterpret_cast<char*>(raw.data()), raw.size());
    if (!file) throw std::runtime_error("Trunacated image file");
    for (std::uint32_t i = 0; i < number_of_images; i++) {
        label(i, raw[i]) = 1;
    }
    file.close();
    Helper::calculate_time(start);
}
std::uint32_t DataReader::bigEndianToHost(std::ifstream& f) {
    unsigned char b[4];
    f.read(reinterpret_cast<char*>(b), 4);
    if (!f) throw std::runtime_error("Unexpected end of file in header");
    return (std::uint32_t{b[0]} << 24) + (std::uint32_t{b[1]} << 16) + (std::uint32_t{b[2]} << 8) + std::uint32_t{b[3]};
}