#include <iostream>
#include <vector>

#include "labelEncoder.hpp"
#include "naiveBayes.hpp"
#include "train-test-split.hpp"

using namespace std;

int main() {

    // Naive Bayes Classifier Example

    vector<vector<string>> data = {
        {"Sunny",    "Hot",  "High",   "Weak",   "No"},
        {"Sunny",    "Hot",  "High",   "Strong", "No"},
        {"Overcast", "Hot",  "High",   "Weak",   "Yes"},
        {"Rain",     "Mild", "High",   "Weak",   "Yes"},
        {"Rain",     "Cool", "Normal", "Weak",   "Yes"},
        {"Rain",     "Cool", "Normal", "Strong", "No"},
        {"Overcast", "Cool", "Normal", "Strong", "Yes"},
        {"Sunny",    "Mild", "High",   "Weak",   "No"},
        {"Sunny",    "Cool", "Normal", "Weak",   "Yes"},
        {"Rain",     "Mild", "Normal", "Weak",   "Yes"},
        {"Sunny",    "Mild", "Normal", "Strong", "Yes"},
        {"Overcast", "Mild", "High",   "Strong", "Yes"},
        {"Overcast", "Hot",  "Normal", "Weak",   "Yes"},
        {"Rain",     "Mild", "High",   "Strong", "No"}
    };


    int n_features = data[0].size();
    vector<LabelEncoder> encoders(n_features);

    vector<vector<string>> columns(n_features);
    for (const auto& row : data) {
        for (int f = 0; f < n_features; f++) {
            columns[f].push_back(row[f]);
        }
    }

    vector<vector<int>> encoded_columns(n_features);
    for (int f = 0; f < n_features; f++) {
        encoded_columns[f] = encoders[f].fit_transform(columns[f]);
    }

    vector<vector<int>> X;
    vector<int> y;
    for (int i = 0; i < data.size(); i++) {
        vector<int> sample;
        for (int f = 0; f < n_features - 1; f++) {
            sample.push_back(encoded_columns[f][i]);
        }
        X.push_back(sample);
        y.push_back(encoded_columns[n_features - 1][i]); 
    }

    NaiveBayes model;
    model.fit(X, y);

    // Predict for {"Sunny", "Cool", "High", "Strong"}
    vector<string> test = {"Sunny", "Cool", "High", "Strong"};
    vector<int> test_encoded;
    for (int f = 0; f < n_features - 1; f++) {
        test_encoded.push_back(encoders[f].fit_transform(test[f])); 
    }

    int pred = model.predict(test_encoded);
    cout << "Predicted Class (encoded): " << pred << endl;
    cout << "Predicted Class (decoded): " << encoders[n_features - 1].inverse_transform(pred) << endl;

    return 0;


}