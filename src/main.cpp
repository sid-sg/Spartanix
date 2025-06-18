#include <iostream>
#include <vector>

#include "naiveBayes.hpp"
#include "train-test-split.hpp"

using namespace std;

int main() {
    vector<vector<int>> X = {{1, 0}, {1, 1}, {0, 0}, {0, 1}, {1, 0}, {0, 0}};
    vector<int> y = {1, 1, 0, 0, 1, 0};

    auto [X_train, X_test, y_train, y_test] = train_test_split(X, y, 0.3, 42);

    NaiveBayes model;
    model.fit(X_train, y_train);

    double acc = model.score(X_test, y_test);
    cout << "Accuracy: " << acc << endl;

    return 0;
}