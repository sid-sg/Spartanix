#include "labelEncoder.hpp"

int LabelEncoder::fit_transform(const std::string& category) {
    if (str2int.find(category) == str2int.end()) {
        str2int[category] = next_label;
        int2str[next_label] = category;
        next_label++;
    }
    return str2int[category];
}

std::vector<int> LabelEncoder::fit_transform(const std::vector<std::string>& categories) {
    std::vector<int> encoded;
    for (const auto& cat : categories) {
        encoded.push_back(fit_transform(cat));
    }
    return encoded;
}

std::string LabelEncoder::inverse_transform(int label) const { return int2str.at(label); }
