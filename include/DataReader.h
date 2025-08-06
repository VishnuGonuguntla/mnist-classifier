#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <vector>

using namespace std;
using Eigen::MatrixXd, Eigen::MatrixXi;
class DataReader {
    public:
    DataReader (string train_data_add, string test_data_add);
    vector<int> src_data_reading(MatrixXd &tr_i, MatrixXd &tr_l, string image, string label, MatrixXi &test_true, string type);
    private:
    int bigToEndian(int i);
};