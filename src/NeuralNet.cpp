
#include <iostream>
#include "NeuralNet.h"

using namespace std;

NN::NN (int source, int test, int n, int b,int k, float eta) {
        train_count = source;
        test_count = test;
        num_classes = k;
        batch_count = b;
        total_count = n;
        learn_rate = eta;
    }

void NN::training(MatrixXd& tr_i,MatrixXd& tr_l, MatrixXd& w1,MatrixXd& w2,int num_epochs,int batch, int hidden_size,float eta) {
    cout<< "training Time: " ;
    auto start= chrono::steady_clock::now();
    MatrixXd s_i(train_count,total_count),a1(batch,hidden_size),a1u=MatrixXd::Ones(batch,hidden_size+1),a2(batch,num_classes),h1(batch,hidden_size),h2(batch,num_classes),h1b(batch,hidden_size+1),h2b(batch,1),h1bu(batch,hidden_size);
    MatrixXd s_l(train_count,num_classes),l(batch,num_classes);
    MatrixXd input(batch,total_count+1),output;
    vector<int> index(train_count,0);
    for (int i = 0; i < train_count; i++) index[i] = i;
    for (int epoch = 0; epoch < 1; epoch++) { // num_epochs
        shuffle_data(tr_i,tr_l,s_i,s_l,index);
        for (int iter = 0; iter < train_count/batch; iter++) { // source/batch
        input(all,seq(0,total_count-1)) = s_i(seq(iter*batch,(iter+1)*batch-1),all); //
        l = s_l(seq(iter*batch,(iter+1)*batch-1),all);
        h1 = input*w1;
        a1 = h1.cwiseMax(0);
        a1u(all,seq(0,hidden_size-1)) = a1;
        h2 = a1u*w2;
        h2 = (h2.array() - h2.maxCoeff());
        h2 = h2.array().exp();
        a2 = h2.array().colwise()/h2.array().rowwise().sum();
        output = -a2.array().log()*l.array();
        // Back:
        h2b = a2.array() - l.array();
        w2 -= eta*a1u.transpose()*h2b;
        a1u = h2*w2.transpose();
        h1b = (a1u.array()>0).cast<double>();
        h1bu(all,all) = h1b(all,seq(0,hidden_size-1));
        w1 -= eta*input.transpose()*h1bu;
        }
        // ofstream file("weight.txt");
        // file << w2 << endl;
        // file.close();
    }
    auto end=chrono::steady_clock::now();
    cout << std::chrono::duration_cast<std::chrono::seconds>(end-start).count() << "\n";
}
    void NN::testing(MatrixXd& w1, MatrixXd& w2, MatrixXd& te_i, MatrixXd& te_l, MatrixXi& true_label,int num_epochs,int batch, int hidden_size,int eta) {
    auto start= chrono::steady_clock::now();
    cout <<"testing time: ";
    MatrixXd h1(test_count,hidden_size),h2(test_count,num_classes),a1u=MatrixXd::Ones(test_count,hidden_size+1),a1(test_count,hidden_size),a2(test_count,num_classes),input=MatrixXd::Ones(test_count,total_count+1);
    input(all,seq(0,train_count-1)) = te_i;
    h1 = input*w1;
    a1 = h1.cwiseMax(0);
    a1u(all,seq(0,hidden_size-1)) = a1;
    h2 = a1u*w2;
    h2 = (h2.array() - h2.maxCoeff());
    h2 = h2.array().exp();
    a2 = h2.array().colwise()/h2.array().rowwise().sum();
    ofstream file2("a2.txt");
    file2 << a2 << endl;
    file2.close();
    int colIndex;
    for (int i=0;i < test_count;i++) {
        if (i%batch==0) cout << "Current batch: " << i/batch<<endl;
        a2.row(i).maxCoeff(&colIndex);
        cout <<" - image "<<i%batch<<": Prediction="<<colIndex<<". Label="<<true_label(i,0)<<endl;
    }
}
void NN::shuffle_data(MatrixXd& tr_i, MatrixXd& tr_l, MatrixXd& s_i, MatrixXd& s_l, vector<int>& index) {
    random_device rd;
    mt19937 gen(rd());
    shuffle(index.begin(), index.end(), gen);
    for (int i = 0; i < train_count; i++) {
        s_i.row(i) = tr_i.row(index[i]);
        s_l.row(i) = tr_l.row(index[i]);
    }
    // ofstream outfile_label("shuffle_image.txt");
    // outfile_label<<s_i<<endl;
    // outfile_label.close();
    // ofstream outfile_laeel("shuffle_label.txt");
    // outfile_laeel<<s_l<<endl;
    // outfile_laeel.close();
}
void NN::initialization(MatrixXd& matrix, string type) {
    int rows = matrix.rows();
    random_device rd;
    mt19937 gen(rd());
    double stddev;
    if (type == "he") {
        stddev = sqrt(2.0/rows);
    } else {
        stddev = sqrt(1.0/rows);
    }
    normal_distribution<double> dist(0,stddev);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < matrix.cols(); j++) {
        matrix(i,j) = dist(gen);
        }
    }
    if (type == "gloriot") {
        matrix = matrix.array().tanh();
    }
}
