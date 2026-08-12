#pragma once

#include "cvrp_instance.h"
#include <vector>

// Evaluate CVRP solution cost
inline double evaluate_cvrp(CVRPSolution& sol, const CVRPInstance& inst) {
    double total_cost = 0.0;

    for (const auto& route : sol.routes) {
        if (route.empty()) continue;

        // Depot -> first customer
        total_cost += inst.distance(0, route[0]);

        // Between customers
        for (size_t i = 0; i < route.size() - 1; i++) {
            total_cost += inst.distance(route[i], route[i + 1]);
        }

        // Last customer -> depot
        total_cost += inst.distance(route.back(), 0);
    }

    sol.cost = total_cost;
    return total_cost;
}

// Check if solution is feasible
inline bool is_feasible(const CVRPSolution& sol, const CVRPInstance& inst) {
    // Check capacity constraints
    for (const auto& route : sol.routes) {
        int load = 0;
        for (int c : route) {
            if (c < 1 || c > inst.n_customers) return false;
            load += inst.demands[c - 1];
        }
        if (load > inst.capacity) return false;
    }

    // Check all customers are served exactly once
    std::vector<bool> served(inst.n_customers, false);
    for (const auto& route : sol.routes) {
        for (int c : route) {
            if (served[c - 1]) return false;  // Duplicate
            served[c - 1] = true;
        }
    }

    for (bool s : served) {
        if (!s) return false;  // Missing customer
    }

    return true;
}
