#include <algorithm>
#include <chrono>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> train_test_split(const std::vector<std::vector<int>>& X, const std::vector<int>& y, double test_size, double random_state = -1) {
    int n_X = X.size();

    if (test_size < 0.0 || test_size > 1.0) {
        throw std::invalid_argument("test_size must be between 0.0 and 1.0");
    }

    std::vector<int> indices(n_X);
    for (int i = 0; i < n_X; i++) indices[i] = i;

    std::mt19937 g;
    if (random_state == -1) {
        g.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    } else {
        g.seed(random_state);
    }

    std::shuffle(indices.begin(), indices.end(), g);

    int test_count = static_cast<int>(n_X * test_size);

    std::vector<std::vector<int>> X_train, X_test;
    std::vector<int> y_train, y_test;

    for (int i = 0; i < n_X; i++) {
        if (i < test_count) {
            X_test.push_back(X[indices[i]]);
            y_test.push_back(y[indices[i]]);
        } else {
            X_train.push_back(X[indices[i]]);
            y_train.push_back(y[indices[i]]);
        }
    }
    return {X_train, X_test, y_train, y_test};
}
