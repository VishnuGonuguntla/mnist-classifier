#include "DataReader.h"

#define MatrixSingle Eigen::MatrixXf
#define MatrixInteger Eigen::MatrixXi

DataReader::DataReader(std::map<std::string, std::string> config_data) {
    train_i = config_data["rel_path_train_images"];
    train_l = config_data["rel_path_train_labels"];
    test_i = config_data["rel_path_test_images"];
    test_l = config_data["rel_path_test_labels"];
}

MatrixSingle* DataReader::parse_image(std::string filename) {
    std::ifstream file;
    file.open(filename, std::ios::binary);

    // Image Data Reading
    if (!file.is_open()) {
        std::cout << "Image Not Found" << std::endl;
        return nullptr;
    }
    int magic_number = 0;
    int number_of_images = 0;
    int number_of_rows = 0;
    int number_of_columns = 0;
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = bigToEndian(magic_number);
    file.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = bigToEndian(number_of_images);
    file.read((char *)&number_of_rows, sizeof(number_of_rows));
    number_of_rows = bigToEndian(number_of_rows);
    file.read((char *)&number_of_columns, sizeof(number_of_columns));
    number_of_columns = bigToEndian(number_of_columns);

    image = MatrixSingle::Zero(number_of_images, number_of_rows * number_of_columns);
    // MatrixXd image(number_of_images, 28*28);
    for (int i = 0; i < number_of_images; i++) {
        for (int j = 0; j < number_of_rows; j++) {
            for (int k = 0; k < number_of_columns; k++) {
                unsigned char value = 0;
                file.read((char *)&value, sizeof(value));
                image(i, k + number_of_columns * j) = value / 255.0f;
            }
        }
    }
    file.close();
    return &image;
}

MatrixInteger* DataReader::parse_label(std::string filename) {
    std::ifstream file;
    file.open(filename, std::ios::binary);
    int magic_number = 0;
    int number_of_images = 0;
    if (!file.is_open()) {
        std::cout << "Label Not open" << std::endl;
        return nullptr;
    }
    file.read((char *)&magic_number, sizeof(magic_number));
    magic_number = bigToEndian(magic_number);
    file.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = bigToEndian(number_of_images);
    label = MatrixInteger::Zero(number_of_images, 10);
    for (int i = 0; i < number_of_images; i++) {
        unsigned char value = 0;
        file.read((char *)&value, sizeof(value));
        label(i, value) = 1;
    }
    file.close();
    return &label;
}
int DataReader::bigToEndian(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}