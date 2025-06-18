#include "naiveBayes.hpp"

#include <cmath>
#include <iostream>

void NaiveBayes::fit(const std::vector<std::vector<int>>& X, const std::vector<int>& y) {
    std::unordered_map<int, int> class_counts;
    int n_samples = X.size();
    int n_features = X[0].size();

    // Prior Probability Calculation
    for (int i = 0; i < n_samples; i++) class_counts[y[i]]++;
    for (auto& c : class_counts) priors[c.first] = static_cast<double>(c.second) / n_samples;

    // Init likelihood structure
    for (auto& c : class_counts) likelihoods[c.first] = std::vector<std::unordered_map<int, double>>(n_features);

    // Count feature values per class
    std::unordered_map<int, std::vector<std::unordered_map<int, int>>> feature_value_counts;

    // Init feature_value_counts structure
    for (auto& c : class_counts) {
        int label = c.first;
        feature_value_counts[label] = std::vector<std::unordered_map<int, int>>(n_features);
    }

    for (int i = 0; i < n_samples; i++) {
        int label = y[i];
        for (int f = 0; f < n_features; f++) {
            int val = X[i][f];
            feature_value_counts[label][f][val]++;
        }
    }

    // Calculate likelihoods with Laplace smoothing
    for (auto& c : class_counts) {
        int label = c.first;
        for (int f = 0; f < n_features; f++) {
            int total_count = 0;
            for (auto& val_count : feature_value_counts[label][f]) total_count += val_count.second;

            int num_unique_values = feature_value_counts[label][f].size();
            for (auto& val_count : feature_value_counts[label][f]) {
                int val = val_count.first;
                int count = val_count.second;
                likelihoods[label][f][val] = (count + 1.0) / (total_count + num_unique_values);  // Laplace smoothing
            }
        }
    }
}

int NaiveBayes::predict(const std::vector<int>& x) {
    double max_prob = -1e9;
    int best_class = -1;

    for (auto& c : priors) {
        int label = c.first;
        double log_prob = log(c.second);  // start with log prior

        for (int f = 0; f < x.size(); f++) {
            double prob = likelihoods[label][f][x[f]];
            if (prob == 0) prob = 1e-6;  // Smoothing fallback
            log_prob += log(prob);
        }

        if (log_prob > max_prob) {
            max_prob = log_prob;
            best_class = label;
        }
    }
    return best_class;
}

double NaiveBayes::score(const std::vector<std::vector<int>>& X, const std::vector<int>& y) {
    int correct = 0;
    for (int i = 0; i < X.size(); i++) {
        if (predict(X[i]) == y[i]) correct++;
    }
    return (double)correct / X.size();
}
