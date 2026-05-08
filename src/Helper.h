#pragma once

#include <fstream>
#include <iostream>
#include <chrono>
#include <map>
#include <random>
#include <string>

#include <Eigen/Dense>

using Eigen::MatrixXd;

class Helper {
public:
    static std::map<std::string, std::string> parse(std::string config_file);
    static void calculate_time(std::chrono::steady_clock::time_point start);
};