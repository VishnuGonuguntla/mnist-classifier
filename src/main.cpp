#include <chrono>
#include <iostream>
#include <map>

#include "DataReader.h"
#include "NeuralNet.h"
#include "Helper.h"

int main(int argc, char *argv[]) {
    std::string config = argv[1];
#ifdef DEBUG
    std::cout << "Debug mode\n";
#endif

    std::map<std::string, std::string> config_data = Helper::parse(config);
    DataReader parser(config_data);

    auto start = std::chrono::steady_clock::now();
    NN network(config_data, "he");
    parser.parse_image(config_data["rel_path_train_images"]);
    parser.parse_label(config_data["rel_path_train_labels"]);
    MatrixSingle& train_images = parser.getImage();
    MatrixInteger& train_labels = parser.getLabel();
    std::cout << train_images.rows() << " " << train_images.cols() << std::endl;
    std::cout << train_labels.rows() << " " << train_labels.cols() << std::endl;
    network.train(train_images, train_labels);

    parser.parse_image(config_data["rel_path_test_images"]);
    parser.parse_label(config_data["rel_path_test_labels"]);
    // network.test();
    auto end = std::chrono::steady_clock::now();
    std::cout <<
    std::chrono::duration_cast<std::chrono::microseconds>(end-start).count()
    << std::endl;
        
    return 0;
}