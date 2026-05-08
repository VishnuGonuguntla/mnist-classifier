#include "NeuralNet.h"

NN::NN(std::map<std::string, std::string> config, std::string initialize) {
    num_epochs = stoi(config["num_epochs"]);
    batch_size = stoi(config["batch_size"]);
    hidden_size = stoi(config["hidden_size"]);
    eta = stof(config["learning_rate"]);
    initializer = initialize;
}


void NN::train(MatrixSingle& image, MatrixSingle& label) {
    std::cout << "training Time: ";
    auto start = std::chrono::steady_clock::now();
    train_count = image.rows();
    num_features = image.cols();
    num_classes = label.cols();
    w1 = MatrixSingle::Zero(num_features + 1, hidden_size);
    w2 = MatrixSingle::Zero(hidden_size + 1, num_classes);
    rng_initialization(w1, initializer);
    rng_initialization(w2, initializer);
    
    Eigen::VectorXi index = Eigen::VectorXi::LinSpaced(train_count, 0, train_count - 1);
    for(int epoch = 0; epoch < num_epochs; epoch++) {
        shuffle_data(index);
        int rows = image.rows();
        int num_batches = rows / batch_size;
        MatrixSingle h1, a1, a2, h2;

        MatrixSingle  images_shuffled = image(index, Eigen::all);
        MatrixSingle labels_shuffled = label(index, Eigen::all);
        MatrixSingle batch_images;
        MatrixSingle batch_labels;
        // MatrixSingle label_float;

        // Then slice contiguous batches cleanly
        for (int b = 0; b < num_batches; b++) {
            // std::cout << "Batch " << b + 1 << "/" << num_batches << std::endl;
            int start = b * batch_size;
            batch_images = images_shuffled(Eigen::seqN(start, batch_size), Eigen::all);
            batch_labels = labels_shuffled(Eigen::seqN(start, batch_size), Eigen::all);
            // label_float = batch_labels.cast<float>();
            
            MatrixSingle batch_images_u(batch_size, num_features + 1);
            batch_images_u << batch_images, MatrixSingle::Ones(batch_size, 1);
            MatrixSingle h2b, h1b, h1bu, output;
            
            h1 = batch_images_u * w1;
            a1 = h1.cwiseMax(0);
            MatrixSingle a1u(batch_size, hidden_size + 1);
            a1u << a1, MatrixSingle::Ones(batch_size, 1);
            h2 = a1u * w2;
            h2 = (h2.array() - h2.maxCoeff());
            h2 = h2.array().exp();
            a2 = h2.array().colwise() / h2.array().rowwise().sum();
            output = -a2.array().log() * batch_labels.array();
            
            // Back:
            h2b = a2.array() - batch_labels.array();
            w2 -= eta * a1u.transpose() * h2b;
            a1u = h2 * w2.transpose();
            h1b = (a1u.array() > 0).cast<float>();
            h1bu = h1b(all, seq(0, hidden_size - 1));
            w1 -= eta * batch_images_u.transpose() * h1bu;
        }
    }

    Helper::calculate_time(start);
    
}
void NN::test(MatrixSingle& image, MatrixSingle& label) {
    auto start = std::chrono::steady_clock::now();
    std::cout << "testing time: ";
    int test_count = image.rows();

    MatrixSingle h1, a1, a2, h2;
    // MatrixSingle label_float = label.cast<float>();


    // for (int i = 0; i < num_batches; i++) {
    MatrixSingle image_u(test_count, num_features +1);
    image_u << image, MatrixSingle::Ones(test_count, 1);
    h1 = image_u * w1;
    a1 = h1.cwiseMax(0);
    MatrixSingle a1u(test_count, hidden_size + 1);
    a1u << a1, MatrixSingle::Ones(test_count, 1);
    h2 = a1u * w2;
    h2 = (h2.array() - h2.maxCoeff());
    h2 = h2.array().exp();
    a2 = h2.array().rowwise() / h2.array().colwise().sum();
    MatrixSingle output = -a2.array().log() * label.array();
    Helper::calculate_time(start);
    Eigen::Index colIndex, colIndex2;
    int count = 0;
    for (int i = 0; i < test_count; i++) {
        output.row(i).maxCoeff(&colIndex);
        if (label(i, colIndex) == 1)
            count++;
        // label.row(i).maxCoeff(&colIndex2);
        // std::cout << "Predicted: " << colIndex << ", Actual: " << colIndex2 << std::endl;
    }
    std::cout << "Correct predictions: " << count << "/" << test_count << std::endl;
    std::cout << "Accuracy: " << (float)count / test_count * 100 << "%" << std::endl;
}

void NN::shuffle_data(Eigen::VectorXi& array) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::shuffle(array.data(), array.data() + array.size(), gen);
}
void NN::rng_initialization(
    MatrixSingle &matrix, std::string type) { // Initialization: Generates the Initial values for the
                        // trainable parameters. Mersenne Twister Engine.
    int rows = matrix.rows() - 1;
    std::random_device rd;
    std::mt19937 gen(rd());
    double stddev;
    if (type == "he") {
        stddev = sqrt(2.0 / rows);
    } else if (type == "gloriot") {
        stddev = sqrt(1.0 / rows);
    }
    std::normal_distribution<double> dist(0, stddev);
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            matrix(i, j) = dist(gen);
        }
    }
}
