#pragma once

#include "cvrp_instance.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

class DataLoader {
public:
    // Load CVRP instances from text format
    // Format:
    // Line 1: num_instances
    // For each instance:
    //   Line: depot_x depot_y
    //   Line: num_customers
    //   Next num_customers lines: customer_x customer_y
    //   Next line: num_customers demand values
    //   Next line: capacity
    static std::vector<CVRPInstance> load_txt(const std::string& txt_path) {
        std::ifstream file(txt_path);
        if (!file) {
            throw std::runtime_error("Cannot open file: " + txt_path);
        }

        std::vector<CVRPInstance> instances;
        int num_instances;
        file >> num_instances;

        for (int idx = 0; idx < num_instances; idx++) {
            double depot_x, depot_y;
            file >> depot_x >> depot_y;
            Point depot(depot_x, depot_y);

            int n_customers;
            file >> n_customers;

            std::vector<Point> customers;
            customers.reserve(n_customers);
            for (int i = 0; i < n_customers; i++) {
                double x, y;
                file >> x >> y;
                customers.emplace_back(x, y);
            }

            std::vector<int> demands;
            demands.reserve(n_customers);
            for (int i = 0; i < n_customers; i++) {
                int d;
                file >> d;
                demands.push_back(d);
            }

            int capacity;
            file >> capacity;

            instances.emplace_back(depot, customers, demands, capacity);
        }

        return instances;
    }
};
