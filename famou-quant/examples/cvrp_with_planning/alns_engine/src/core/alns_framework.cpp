#include "alns_framework.h"
#include "cvrp_evaluate.h"
#include <chrono>
#include <iostream>
#include <iomanip>
#include <algorithm>

ALNSFramework::ALNSFramework(const CVRPInstance& inst,
                             const std::vector<std::pair<std::string, DestroyFunc>>& destroy,
                             const std::vector<std::pair<std::string, RepairFunc>>& repair,
                             LocalSearchFunc ls_func,
                             double t_start, double t_end, double rate,
                             int seg_size, int sd, bool verb)
    : instance(inst),
      destroy_ops(destroy),
      repair_ops(repair),
      local_search_func(ls_func),
      T_start(t_start),
      T_end(t_end),
      cooling_rate(rate),
      segment_size(seg_size),
      seed(sd),
      verbose(verb),
      sa(t_start, t_end, rate),
      destroy_weights(get_op_names(destroy), 33.0, 9.0, 13.0, 0.8),
      repair_weights(get_op_names(repair), 33.0, 9.0, 13.0, 0.8),
      rng(sd),
      iters(0),
      best_cost(1e9) {}

CVRPSolution ALNSFramework::solve(CVRPSolution& initial_solution, double time_limit) {
    auto start_time = std::chrono::steady_clock::now();

    CVRPSolution current = initial_solution.copy();
    CVRPSolution best = initial_solution.copy();

    evaluate_cvrp(current, instance);
    evaluate_cvrp(best, instance);

    double current_cost = current.cost;
    best_cost = best.cost;

    if (verbose) {
        std::cout << "Initial cost: " << std::fixed << std::setprecision(4)
                  << best_cost << std::endl;
    }

    iters = 0;
    int max_iters = 100000;  // Default max iterations

    while (iters < max_iters) {
        // Check time limit
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        if (elapsed >= time_limit) break;

        // Select operators
        std::string destroy_name = destroy_weights.select(rng);
        std::string repair_name = repair_weights.select(rng);

        // Find the operator functions
        DestroyFunc destroy_op = nullptr;
        RepairFunc repair_op = nullptr;

        for (const auto& pair : destroy_ops) {
            if (pair.first == destroy_name) {
                destroy_op = pair.second;
                break;
            }
        }

        for (const auto& pair : repair_ops) {
            if (pair.first == repair_name) {
                repair_op = pair.second;
                break;
            }
        }

        if (!destroy_op || !repair_op) {
            std::cerr << "Error: operator not found!" << std::endl;
            break;
        }

        // Destroy
        int n_remove = get_n_remove(rng);
        CVRPSolution destroyed = destroy_op(current, n_remove, instance, rng);

        // Get removed customers
        std::vector<int> removed = get_removed_customers(current, destroyed, instance);

        // Repair
        CVRPSolution neighbor = repair_op(destroyed, removed, instance);
        evaluate_cvrp(neighbor, instance);

        // Apply local search if provided
        if (local_search_func) {
            local_search_func(neighbor, instance);
        }

        double neighbor_cost = neighbor.cost;

        // Determine outcome
        int outcome = 4;  // Rejected
        bool accepted = false;

        if (neighbor_cost < best_cost) {
            // New best
            best = neighbor.copy();
            best_cost = neighbor_cost;
            current = neighbor.copy();
            current_cost = neighbor_cost;
            outcome = 1;
            accepted = true;

            if (verbose && iters % 100 == 0) {
                std::cout << "Iter " << std::setw(6) << iters
                          << " | Best: " << std::fixed << std::setprecision(4) << best_cost
                          << " | T: " << std::setprecision(2) << sa.T
                          << std::endl;
            }
        } else if (neighbor_cost < current_cost) {
            // Better than current
            current = neighbor.copy();
            current_cost = neighbor_cost;
            outcome = 3;
            accepted = true;
        } else if (sa.accept(current_cost, neighbor_cost, rng)) {
            // Accepted by SA
            current = neighbor.copy();
            current_cost = neighbor_cost;
            outcome = 2;
            accepted = true;
        } else {
            // Rejected
            outcome = 4;
        }

        // Update weights
        destroy_weights.update(destroy_name, outcome);
        repair_weights.update(repair_name, outcome);

        // Cool down
        sa.cool();

        iters++;

        // Update weights every segment
        if (iters % segment_size == 0) {
            destroy_weights.update_weights();
            repair_weights.update_weights();
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(end_time - start_time).count();

    if (verbose) {
        std::cout << "\nFinished: " << iters << " iters, "
                  << std::fixed << std::setprecision(1) << total_time << "s"
                  << std::endl;
        std::cout << "Best cost: " << std::setprecision(4) << best_cost << std::endl;
        std::cout << "Iter/s: " << std::setprecision(2) << (iters / total_time) << std::endl;
    }

    return best;
}

int ALNSFramework::get_n_remove(std::mt19937& rng) {
    // Remove 10-40% of customers
    int min_remove = std::max(1, instance.n_customers / 10);
    int max_remove = std::max(min_remove, instance.n_customers * 4 / 10);
    std::uniform_int_distribution<> dist(min_remove, max_remove);
    return dist(rng);
}

std::vector<std::string> ALNSFramework::get_op_names(
    const std::vector<std::pair<std::string, DestroyFunc>>& ops) {
    std::vector<std::string> names;
    for (const auto& pair : ops) {
        names.push_back(pair.first);
    }
    return names;
}

std::vector<std::string> ALNSFramework::get_op_names(
    const std::vector<std::pair<std::string, RepairFunc>>& ops) {
    std::vector<std::string> names;
    for (const auto& pair : ops) {
        names.push_back(pair.first);
    }
    return names;
}
