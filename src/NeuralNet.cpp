#include "NeuralNet.h"

NN::NN(std::map<std::string, std::string> config, std::string initialize) {
    num_epochs = stoi(config["num_epochs"]);
    batch_size = stoi(config["batch_size"]);
    hidden_size = stoi(config["hidden_size"]);
    eta = stof(config["learning_rate"]);
    initializer = initialize;
    num_classes = 10;
    test_count = 10000;

}


void NN::train(MatrixSingle& image, MatrixInteger& label) {
    std::cout << "training Time: ";
    auto start = std::chrono::steady_clock::now();
    train_count = image.rows();
    int rows = image.rows();
    int cols = image.cols();
    w1 = MatrixSingle::Zero(cols + 1, hidden_size);
    w2 = MatrixSingle::Zero(hidden_size + 1, num_classes);
    rng_initialization(w1, initializer);
    rng_initialization(w2, initializer);
    
    Eigen::VectorXi index = Eigen::VectorXi::LinSpaced(train_count, 0, train_count - 1);
    for(int epoch = 0; epoch < num_epochs; epoch++) {
        // std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << std::endl;
        shuffle_data(index);
        int rows = image.rows();
        int cols = image.cols();
        int num_batches = rows / batch_size;
        MatrixSingle h1, a1, a2, h2;

        // MatrixSingle batch_images(batch_size, image.cols() + 1);
        // MatrixInteger batch_labels(batch_size, 10);
        
        MatrixSingle  images_shuffled = image(index, Eigen::all);
        MatrixInteger labels_shuffled = label(index, Eigen::all);
        MatrixSingle batch_images;
        MatrixInteger batch_labels;
        MatrixSingle label_float;

        // Then slice contiguous batches cleanly
        for (int b = 0; b < num_batches; b++) {
            // std::cout << "Batch " << b + 1 << "/" << num_batches << std::endl;
            int start = b * batch_size;
            batch_images = images_shuffled(Eigen::seqN(start, batch_size), Eigen::all);
            batch_labels = labels_shuffled(Eigen::seqN(start, batch_size), Eigen::all);
            label_float = batch_labels.cast<float>();
            
            MatrixSingle batch_images_u(batch_size, cols + 1);
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
            output = -a2.array().log() * label_float.array();
            
            // Back:
            h2b = a2.array() - label_float.array();
            w2 -= eta * a1u.transpose() * h2b;
            a1u = h2 * w2.transpose();
            h1b = (a1u.array() > 0).cast<float>();
            h1bu = h1b(all, seq(0, hidden_size - 1));
            w1 -= eta * batch_images_u.transpose() * h1bu;
        }
    }

    
    auto end = std::chrono::steady_clock::now();
    std::cout
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << std::endl;
}
void NN::test(MatrixSingle& image, MatrixInteger& label) {
    auto start = std::chrono::steady_clock::now();
    std::cout << "testing time: ";
    int rows = image.rows();
    int cols = image.cols();
    // input(all, seq(0, train_count - 1)) = image;

    // std::vector<int> index(test_count, 0);
    // std::iota(index.begin(), index.end(), 0);
    // shuffle_data(index);
    // int num_batches = rows / batch_size;
    MatrixSingle h1, a1, a2, h2;
    MatrixSingle label_float = label.cast<float>();


    // for (int i = 0; i < num_batches; i++) {
    MatrixSingle image_u(rows, cols +1);
    image_u << image, MatrixSingle::Ones(rows, 1);
    h1 = image_u * w1;
    a1 = h1.cwiseMax(0);
    MatrixSingle a1u(test_count, hidden_size + 1);
    a1u << a1, MatrixSingle::Ones(test_count, 1);
    h2 = a1u * w2;
    h2 = (h2.array() - h2.maxCoeff());
    h2 = h2.array().exp();
    a2 = h2.array().rowwise() / h2.array().colwise().sum();
    MatrixSingle output = -a2.array().log() * label_float.array();
    auto end = std::chrono::steady_clock::now();
    std::cout
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << std::endl;
    std::ofstream file2("a3.txt");
    file2 << output << std::endl;
    file2.close();
    // int colIndex;
    // for (int i = 0; i < test_count; i++) {
    //     if (i % batch_size == 0)
    //         std::cout << "Current batch: " << i / batch_size << std::endl;
    //     a2.row(i).maxCoeff(&colIndex);
    //     std::cout << " - image " << i % batch_size
    //             << ": Prediction=" << colIndex
    //             << ". Label=" << label(i, colIndex) << std::endl;
    // }
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
