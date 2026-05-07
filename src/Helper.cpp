#include "Helper.h"

std::map<std::string, std::string> Helper::parse(std::string config_file) {
    std::map<std::string, std::string> parse_data;
    std::ifstream file(config_file);
    std::string line;
    if (!file.is_open()) {
        std::cout << "Error opening file " << config_file << std::endl;
        return parse_data;
    }
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        size_t pos = line.find_first_of(" = ");
        int i = 0;
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 3);
            parse_data[key] = value;
        }
    }
    return parse_data;
}