#pragma once

#include <vector>
#include <map>
#include <string>
#include <random>
#include <cmath>
#include <functional>
#include "cvrp_instance.h"

// Type aliases for operators (used by alns_operators.h)
using DestroyFunc = std::function<CVRPSolution(const CVRPSolution&, int, const CVRPInstance&, std::mt19937&)>;
using RepairFunc = std::function<CVRPSolution(CVRPSolution&, const std::vector<int>&, const CVRPInstance&)>;
using InitFunc = std::function<CVRPSolution(const CVRPInstance&, std::mt19937&)>;
using LocalSearchFunc = std::function<void(CVRPSolution&, const CVRPInstance&)>;

// Helper function to get removed customers
inline std::vector<int> get_removed_customers(const CVRPSolution& original,
                                               const CVRPSolution& destroyed,
                                               const CVRPInstance& inst) {
    std::vector<bool> in_destroyed(inst.n_customers + 1, false);

    for (const auto& route : destroyed.routes) {
        for (int c : route) {
            in_destroyed[c] = true;
        }
    }

    std::vector<int> removed;
    for (int i = 1; i <= inst.n_customers; i++) {
        if (!in_destroyed[i]) {
            removed.push_back(i);
        }
    }

    return removed;
}

// Simulated Annealing acceptance
class SimulatedAnnealing {
public:
    double T;
    double T_start;
    double T_end;
    double cooling_rate;

    SimulatedAnnealing(double t_start = 100.0, double t_end = 0.1, double rate = 0.995)
        : T(t_start), T_start(t_start), T_end(t_end), cooling_rate(rate) {}

    bool accept(double current_cost, double new_cost, std::mt19937& rng) {
        if (new_cost < current_cost) return true;

        double delta = new_cost - current_cost;
        double prob = std::exp(-delta / T);

        std::uniform_real_distribution<> dist(0.0, 1.0);
        return dist(rng) < prob;
    }

    void cool() {
        T = std::max(T_end, T * cooling_rate);
    }

    void reset() {
        T = T_start;
    }
};

// Adaptive operator weights
class AdaptiveWeights {
public:
    std::map<std::string, double> weights;
    std::map<std::string, double> scores;
    std::map<std::string, int> counts;

    double sigma1;  // New best
    double sigma2;  // Accepted
    double sigma3;  // Better but not accepted
    double decay;

    AdaptiveWeights(const std::vector<std::string>& operators,
                   double s1 = 33.0, double s2 = 9.0, double s3 = 13.0, double d = 0.8)
        : sigma1(s1), sigma2(s2), sigma3(s3), decay(d) {
        for (const auto& op : operators) {
            weights[op] = 1.0;
            scores[op] = 0.0;
            counts[op] = 0;
        }
    }

    std::string select(std::mt19937& rng) {
        // Build discrete distribution
        std::vector<std::string> ops;
        std::vector<double> ws;

        for (const auto& pair : weights) {
            ops.push_back(pair.first);
            ws.push_back(pair.second);
        }

        std::discrete_distribution<> dist(ws.begin(), ws.end());
        int idx = dist(rng);
        return ops[idx];
    }

    void update(const std::string& op, int outcome) {
        // outcome: 1=new_best, 2=accepted, 3=better, 4=rejected
        double score = 0.0;
        if (outcome == 1) score = sigma1;
        else if (outcome == 2) score = sigma2;
        else if (outcome == 3) score = sigma3;

        scores[op] += score;
        counts[op]++;
    }

    void update_weights() {
        for (auto& pair : weights) {
            const std::string& op = pair.first;
            if (counts[op] > 0) {
                weights[op] = decay * weights[op] + (1.0 - decay) * (scores[op] / counts[op]);
            }

            // Reset counters
            scores[op] = 0.0;
            counts[op] = 0;
        }
    }
};

// Main ALNS Framework class
class ALNSFramework {
public:
    const CVRPInstance& instance;

    // Operators (registered from alns_operators.h)
    std::vector<std::pair<std::string, DestroyFunc>> destroy_ops;
    std::vector<std::pair<std::string, RepairFunc>> repair_ops;
    LocalSearchFunc local_search_func;

    // Parameters
    double T_start;
    double T_end;
    double cooling_rate;
    int segment_size;
    int seed;
    bool verbose;

    // State
    SimulatedAnnealing sa;
    AdaptiveWeights destroy_weights;
    AdaptiveWeights repair_weights;
    std::mt19937 rng;

    // Stats
    int iters;
    double best_cost;

    ALNSFramework(const CVRPInstance& inst,
                  const std::vector<std::pair<std::string, DestroyFunc>>& destroy,
                  const std::vector<std::pair<std::string, RepairFunc>>& repair,
                  LocalSearchFunc ls_func = nullptr,
                  double t_start = 500.0, double t_end = 0.01, double rate = 0.9995,
                  int seg_size = 100, int sd = 42, bool verb = false);

    // Run ALNS (implementation in .cpp)
    CVRPSolution solve(CVRPSolution& initial_solution, double time_limit);

private:
    int get_n_remove(std::mt19937& rng);
    std::vector<std::string> get_op_names(const std::vector<std::pair<std::string, DestroyFunc>>& ops);
    std::vector<std::string> get_op_names(const std::vector<std::pair<std::string, RepairFunc>>& ops);
};
