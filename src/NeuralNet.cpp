#include "NeuralNet.h"

NN::NN(std::map<std::string, std::string> config, std::string initialize) {
    num_epochs = stoi(config["num_epochs"]);
    batch_size = stoi(config["batch_size"]);
    hidden_size = stoi(config["hidden_size"]);
    eta = stof(config["learning_rate"]);
    initializer = initialize;

}

void NN::forward(MatrixSingle& image, MatrixInteger &label, Eigen::VectorXi& idx) {
    int rows = image.rows();
    int cols = image.cols();
    int num_batches = rows / batch_size;

    // MatrixSingle batch_images(batch_size, image.cols() + 1);
    // MatrixInteger batch_labels(batch_size, 10);
    
    MatrixSingle  images_shuffled = image(idx, Eigen::all);
    MatrixInteger labels_shuffled = label(idx, Eigen::all);
    MatrixSingle batch_images;
    MatrixInteger batch_labels;

    // Then slice contiguous batches cleanly
    for (int b = 0; b < num_batches; b++) {
        std::cout << "Batch " << b + 1 << "/" << num_batches << std::endl;
        int start = b * batch_size;
        batch_images = images_shuffled(Eigen::seqN(start, batch_size), Eigen::all);
        batch_labels = labels_shuffled(Eigen::seqN(start, batch_size), Eigen::all);
        
        MatrixSingle batch_images_u(batch_size, cols + 1);
        batch_images_u << batch_images, MatrixSingle::Ones(batch_size, 1);
        
        h1 = batch_images_u * w1;
        a1 = h1.cwiseMax(0);
        MatrixSingle a1u(batch_size, hidden_size + 1);
        a1u << a1, MatrixSingle::Ones(batch_size, 1);
        h2 = a1u * w2;
        h2 = (h2.array() - h2.maxCoeff());
        h2 = h2.array().exp();
        a2 = h2.array().colwise() / h2.array().rowwise().sum();
    }

}

// void NN::backward(MatrixS) {

// }
void NN::train(MatrixSingle& tr_i, MatrixInteger& tr_l) {
    std::cout << "training Time: ";
    auto start = std::chrono::steady_clock::now();
    train_count = tr_i.rows();
    int rows = tr_i.rows();
    int cols = tr_i.cols();
    w1 = MatrixSingle::Zero(cols + 1, hidden_size);
    w2 = MatrixSingle::Zero(hidden_size + 1, num_classes);
    rng_initialization(w1, initializer);
    rng_initialization(w2, initializer);
    
    Eigen::VectorXi index = Eigen::VectorXi::LinSpaced(train_count, 0, train_count - 1);
    for (int i = 0; i <= batch_size; i++) {
        std::cout << w1(i, 0) << " " << w1(i, 1) << " " << w1(i, 2) << std::endl;
    }
    for(int epoch = 0; epoch < 1; epoch++) {
        shuffle_data(index);
        forward(tr_i, tr_l, index);

        
    }
    for (int i = 0; i <= batch_size; i++) {
        std::cout << w1(i, 0) << " " << w1(i, 1) << " " << w1(i, 2) << std::endl;
    }

    // MatrixSingle s_i(train_count, total_count), a1(batch_size, hidden_size),
    //     a1u = MatrixSingle::Ones(batch_size, hidden_size + 1),
    //     a2(batch_size, num_classes), h1(batch_size, hidden_size),
    //     h2(batch_size, num_classes), h1b(batch_size, hidden_size + 1),
    //     h2b(batch_size, 1), h1bu(batch_size, hidden_size);
    // MatrixSingle s_l(train_count, num_classes), l(batch_size, num_classes);
    // MatrixSingle input(batch_size, total_count + 1), output;

    
    // for (int epoch = 0; epoch < num_epochs; epoch++) { // num_epochs
    //     shuffle_data(index);
    //     for (int iter = 0; iter < train_count / batch; iter++) { // source/batch
    //         input(Eigen::all, seq(0, total_count - 1)) =
    //             s_i(seq(iter * batch, (iter + 1) * batch - 1), all); //
    //         l = s_l(seq(iter * batch, (iter + 1) * batch - 1), all);
    //         h1 = input * w1;
    //         a1 = h1.cwiseMax(0);
    //         a1u(all, seq(0, hidden_size - 1)) = a1;
    //         h2 = a1u * w2;
    //         h2 = (h2.array() - h2.maxCoeff());
    //         h2 = h2.array().exp();
    //         a2 = h2.array().colwise() / h2.array().rowwise().sum();
    //         output = -a2.array().log() * l.array();
    //         // Back:
    //         h2b = a2.array() - l.array();
    //         w2 -= eta * a1u.transpose() * h2b;
    //         a1u = h2 * w2.transpose();
    //         h1b = (a1u.array() > 0).cast<double>();
    //         h1bu(all, all) = h1b(all, seq(0, hidden_size - 1));
    //         w1 -= eta * input.transpose() * h1bu;
    //     }
    //     // std::ofstream file("weight.txt");
    //     // file << w2 << std::endl;
    //     // file.close();
    // }
    auto end = std::chrono::steady_clock::now();
    std::cout
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << std::endl;
}
void NN::test(MatrixSingle& te_i, MatrixInteger& true_label) {
    auto start = std::chrono::steady_clock::now();
    std::cout << "testing time: ";
    MatrixSingle h1(test_count, hidden_size), h2(test_count, num_classes),
        a1u = MatrixSingle::Ones(test_count, hidden_size + 1),
        a1(test_count, hidden_size), a2(test_count, num_classes),
        input = MatrixSingle::Ones(test_count, total_count + 1);
    input(all, seq(0, train_count - 1)) = te_i;

    // std::vector<int> index(test_count, 0);
    // std::iota(index.begin(), index.end(), 0);
    // shuffle_data(index);
    h1 = input * w1;
    a1 = h1.cwiseMax(0);
    a1u(all, seq(0, hidden_size - 1)) = a1;
    h2 = a1u * w2;
    h2 = (h2.array() - h2.maxCoeff());
    h2 = h2.array().exp();
    a2 = h2.array().colwise() / h2.array().rowwise().sum();
    std::ofstream file2("a2.txt");
    file2 << a2 << std::endl;
    file2.close();
    int colIndex;
    for (int i = 0; i < test_count; i++) {
        if (i % batch_size == 0)
            std::cout << "Current batch: " << i / batch_size << std::endl;
        a2.row(i).maxCoeff(&colIndex);
        std::cout << " - image " << i % batch_size
                  << ": Prediction=" << colIndex
                  << ". Label=" << true_label(i, 0) << std::endl;
    }
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
