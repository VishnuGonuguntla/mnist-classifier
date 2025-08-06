#include <Eigen/Dense>
#include <string>
#include <random>
#include <map>


using Eigen::MatrixXd;
using namespace std;

class Helper {
    public:
    void rng_initialization(MatrixXd& matrix,string type) ;

    map<string,string> parse(string config_file);
};