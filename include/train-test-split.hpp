#pragma once

#include <tuple>
#include <vector>

std::tuple<std::vector<std::vector<int>>, std::vector<std::vector<int>>, std::vector<int>, std::vector<int>> train_test_split(const std::vector<std::vector<int>>& X, const std::vector<int>& y, double test_size, double random_state);
