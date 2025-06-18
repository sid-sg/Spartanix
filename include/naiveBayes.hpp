#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>

class NaiveBayes {
    std::unordered_map<int, double> priors; 
    std::unordered_map<int, std::vector<std::unordered_map<int, double>>> likelihoods;

public:
    void fit(const std::vector<std::vector<int>>& X, const std::vector<int>& y);

    int predict(const std::vector<int>& x);

    double score(const std::vector<std::vector<int>>& X, const std::vector<int>& y);
};
