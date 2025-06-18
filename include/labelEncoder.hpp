#pragma once

#include <string>
#include <unordered_map>
#include <vector>

class LabelEncoder {
    std::unordered_map<std::string, int> str2int;
    std::unordered_map<int, std::string> int2str;
    int next_label = 0;

   public:
    int fit_transform(const std::string& category);

    std::vector<int> fit_transform(const std::vector<std::string>& categories);

    std::string inverse_transform(int label) const;
};
