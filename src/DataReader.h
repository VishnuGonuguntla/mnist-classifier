#pragma once

#include <string>
#include <map>
#include <fstream>


#include <Eigen/Dense>
#include "Helper.h"

using MatrixSingle = Eigen::MatrixXf;
using MatrixInteger = Eigen::MatrixXi;

class DataReader {
    public:
    DataReader() = default;
    explicit DataReader (const std::map<std::string,std::string>& config_data);
    void parse_image(const std::string& filename);
    void parse_label(const std::string& filename);
    const MatrixSingle& getImage() const { return image; }
    const MatrixSingle& getLabel() const { return label; }
    private:
    static std::uint32_t bigEndianToHost(std::ifstream& f);

    MatrixSingle image;
    MatrixSingle label;
    std::string train_i, train_l, test_i, test_l;
};