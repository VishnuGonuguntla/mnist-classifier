
#include "NeuralNet.h"

NN::NN(std::map<std::string, std::string> config, std::string initializer) {
    num_epochs = stoi(config["num_epochs"]);
    batch_size = stoi(config["batch_size"]);
    hidden_size = stoi(config["hidden_size"]);
    eta = stof(config["learning_rate"]);
    rng_initialization(w1, initializer);
    rng_initialization(w2, initializer);
}

void NN::forward() {}

void NN::backward() {}
void NN::train() {
    std::cout << "training Time: ";
    auto start = std::chrono::steady_clock::now();
    MatrixSingle s_i(train_count, total_count), a1(batch_size, hidden_size),
        a1u = MatrixSingle::Ones(batch_size, hidden_size + 1),
        a2(batch_size, num_classes), h1(batch_size, hidden_size),
        h2(batch_size, num_classes), h1b(batch_size, hidden_size + 1),
        h2b(batch_size, 1), h1bu(batch_size, hidden_size);
    MatrixSingle s_l(train_count, num_classes), l(batch_size, num_classes);
    MatrixSingle input(batch_size, total_count + 1), output;
    std::vector<int> index(train_count, 0);
    for (int i = 0; i < train_count; i++)
        index[i] = i;
    for (int epoch = 0; epoch < num_epochs; epoch++) { // num_epochs
        shuffle_data(tr_i, tr_l, s_i, s_l, index);
        for (int iter = 0; iter < train_count / batch; iter++) { // source/batch
            input(Eigen::all, seq(0, total_count - 1)) =
                s_i(seq(iter * batch, (iter + 1) * batch - 1), all); //
            l = s_l(seq(iter * batch, (iter + 1) * batch - 1), all);
            h1 = input * w1;
            a1 = h1.cwiseMax(0);
            a1u(all, seq(0, hidden_size - 1)) = a1;
            h2 = a1u * w2;
            h2 = (h2.array() - h2.maxCoeff());
            h2 = h2.array().exp();
            a2 = h2.array().colwise() / h2.array().rowwise().sum();
            output = -a2.array().log() * l.array();
            // Back:
            h2b = a2.array() - l.array();
            w2 -= eta * a1u.transpose() * h2b;
            a1u = h2 * w2.transpose();
            h1b = (a1u.array() > 0).cast<double>();
            h1bu(all, all) = h1b(all, seq(0, hidden_size - 1));
            w1 -= eta * input.transpose() * h1bu;
        }
        // std::ofstream file("weight.txt");
        // file << w2 << std::endl;
        // file.close();
    }
    auto end = std::chrono::steady_clock::now();
    std::cout
        << std::chrono::duration_cast<std::chrono::seconds>(end - start).count()
        << "\n";
}
void NN::test() {
    auto start = std::chrono::steady_clock::now();
    std::cout << "testing time: ";
    MatrixSingle h1(test_count, hidden_size), h2(test_count, num_classes),
        a1u = MatrixSingle::Ones(test_count, hidden_size + 1),
        a1(test_count, hidden_size), a2(test_count, num_classes),
        input = MatrixSingle::Ones(test_count, total_count + 1);
    input(all, seq(0, train_count - 1)) = te_i;
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
void NN::shuffle_data(MatrixSingle &tr_i, MatrixSingle &tr_l, MatrixSingle &s_i,
                      MatrixSingle &s_l, std::vector<int> &index) {
    // Should return a shuffled indices vector
    // called during each epoch
    std::random_device rd;
    std::mt19937 gen(rd());
    shuffle(index.begin(), index.end(), gen);
    for (int i = 0; i < train_count; i++) {
        s_i.row(i) = tr_i.row(index[i]);
        s_l.row(i) = tr_l.row(index[i]);
        parser.parse_image(config_data["rel_path_train_images"]);
    }
    // std::ofstream outfile_label("shuffle_image.txt");
    // outfile_label<<s_i<<std::endl;
    // outfile_label.close();
    // std::ofstream outfile_laeel("shuffle_label.txt");
    // outfile_laeel<<s_l<<std::endl;
    // outfile_laeel.close();
}

void NN::rng_initialization(
    MatrixSingle &matrix,
    std::string type) { // Initialization: Generates the Initial values for the
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
