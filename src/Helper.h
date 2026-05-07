#pragma once

#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <string>

#include <Eigen/Dense>

using Eigen::MatrixXd;

class Helper {
  public:
    void rng_initialization(MatrixXd &matrix, std::string type);

    static std::map<std::string, std::string> parse(std::string config_file);
};