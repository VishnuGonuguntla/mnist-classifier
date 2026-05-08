#include <chrono>
#include <iostream>
#include <map>

#include "DataReader.h"
#include "NeuralNet.h"
#include "Helper.h"

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return 1;
    }
    std::string config = argv[1];
#ifdef DEBUG
    std::cout << "Debug mode\n";
#endif

    std::map<std::string, std::string> config_data = Helper::parse(config);
    DataReader parser(config_data);

    NN network(config_data, "he");

    parser.parse_image(config_data["rel_path_train_images"]);
    parser.parse_label(config_data["rel_path_train_labels"]);
    MatrixSingle& train_images = parser.getImage();
    MatrixSingle& train_labels = parser.getLabel();
    network.train(train_images, train_labels);

    parser.parse_image(config_data["rel_path_test_images"]);
    parser.parse_label(config_data["rel_path_test_labels"]);
    MatrixSingle& test_images = parser.getImage();
    MatrixSingle& test_labels = parser.getLabel();
    network.test(test_images, test_labels);
            
    return 0;
}