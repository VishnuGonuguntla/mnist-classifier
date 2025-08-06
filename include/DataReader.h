#include <iostream>
#include <fstream>
#include <string>
#include <Eigen/Dense>
#include <vector>
#include <map>

using namespace std;
using Eigen::MatrixXd, Eigen::MatrixXi;

class DataReader {
    public:
    DataReader (map<string,string> config_data);
    vector<int> deCompress(MatrixXd& image, MatrixXi& label, string type);
    private:
    string train_i, train_l, test_i, test_l;

    int bigToEndian(int i);
};