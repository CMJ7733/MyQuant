#pragma once

#include "core/cvrp_instance.h"
#include "core/cvrp_evaluate.h"
#include "core/alns_framework.h"
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>
#include <limits>

//==============================================================================
// BLOCK A START: DESTROY OPERATORS
//==============================================================================
// FM can modify any function in this block, or add new destroy operators

// Random removal
inline CVRPSolution destroy_random(const CVRPSolution& sol, int n_remove,
                                   const CVRPInstance& inst, std::mt19937& rng) {
    CVRPSolution new_sol = sol.copy();
    
    // Collect all customers
    std::vector<int> all_customers;
    for (const auto& route : new_sol.routes) {
        all_customers.insert(all_customers.end(), route.begin(), route.end());
    }
    
    if (all_customers.empty()) return new_sol;
    
    // Randomly select customers to remove
    std::shuffle(all_customers.begin(), all_customers.end(), rng);
    int to_remove = std::min(n_remove, (int)all_customers.size());
    
    std::unordered_set<int> removed_set(all_customers.begin(),
                                       all_customers.begin() + to_remove);
    
    // Remove from routes
    for (size_t i = 0; i < new_sol.routes.size(); i++) {
        auto& route = new_sol.routes[i];
        route.erase(
            std::remove_if(route.begin(), route.end(),
                          [&](int c) { return removed_set.count(c) > 0; }),
            route.end()
        );
        new_sol.update_route_load(i, inst);
    }
    
    new_sol.remove_empty_routes();
    return new_sol;
}

// Worst removal - remove customers with highest distance to neighbors
inline CVRPSolution destroy_worst(const CVRPSolution& sol, int n_remove,
                                  const CVRPInstance& inst, std::mt19937& rng) {
    CVRPSolution new_sol = sol.copy();
    
    struct CustomerCost {
        int customer;
        double cost;
        int route_idx;
        int pos;
    };
    
    std::vector<CustomerCost> costs;
    
    // Calculate removal cost for each customer
    for (size_t ri = 0; ri < new_sol.routes.size(); ri++) {
        const auto& route = new_sol.routes[ri];
        for (size_t i = 0; i < route.size(); i++) {
            int c = route[i];
            int prev = (i == 0) ? 0 : route[i - 1];
            int next = (i == route.size() - 1) ? 0 : route[i + 1];
            
            double cost_before = inst.distance(prev, c) + inst.distance(c, next);
            double cost_after = inst.distance(prev, next);
            double saving = cost_before - cost_after;
            
            costs.push_back({c, -saving, (int)ri, (int)i});
        }
    }
    
    if (costs.empty()) return new_sol;
    
    // Sort by cost (descending)
    std::sort(costs.begin(), costs.end(),
             [](const CustomerCost& a, const CustomerCost& b) { return a.cost > b.cost; });
    
    // Remove worst customers
    int to_remove = std::min(n_remove, (int)costs.size());
    std::unordered_set<int> removed_set;
    for (int i = 0; i < to_remove; i++) {
        removed_set.insert(costs[i].customer);
    }
    
    // Remove from routes
    for (size_t i = 0; i < new_sol.routes.size(); i++) {
        auto& route = new_sol.routes[i];
        route.erase(
            std::remove_if(route.begin(), route.end(),
                          [&](int c) { return removed_set.count(c) > 0; }),
            route.end()
        );
        new_sol.update_route_load(i, inst);
    }
    
    new_sol.remove_empty_routes();
    return new_sol;
}

// Shaw removal - remove similar customers
inline CVRPSolution destroy_shaw(const CVRPSolution& sol, int n_remove,
                                const CVRPInstance& inst, std::mt19937& rng) {
    CVRPSolution new_sol = sol.copy();
    
    // Collect all customers
    std::vector<int> all_customers;
    for (const auto& route : new_sol.routes) {
        all_customers.insert(all_customers.end(), route.begin(), route.end());
    }
    
    if (all_customers.empty()) return new_sol;
    
    // Select random seed customer
    std::uniform_int_distribution<> dist(0, all_customers.size() - 1);
    int seed_customer = all_customers[dist(rng)];
    
    // Calculate similarity (using distance as measure)
    struct Similarity {
        int customer;
        double distance;
    };
    
    std::vector<Similarity> similarities;
    for (int c : all_customers) {
        if (c == seed_customer) continue;
        double d = inst.distance(seed_customer, c);
        similarities.push_back({c, d});
    }
    
    // Sort by similarity (closest first)
    std::sort(similarities.begin(), similarities.end(),
             [](const Similarity& a, const Similarity& b) { return a.distance < b.distance; });
    
    // Remove seed + most similar customers
    std::unordered_set<int> removed_set;
    removed_set.insert(seed_customer);
    int to_remove = std::min(n_remove - 1, (int)similarities.size());
    for (int i = 0; i < to_remove; i++) {
        removed_set.insert(similarities[i].customer);
    }
    
    // Remove from routes
    for (size_t i = 0; i < new_sol.routes.size(); i++) {
        auto& route = new_sol.routes[i];
        route.erase(
            std::remove_if(route.begin(), route.end(),
                          [&](int c) { return removed_set.count(c) > 0; }),
            route.end()
        );
        new_sol.update_route_load(i, inst);
    }
    
    new_sol.remove_empty_routes();
    return new_sol;
}

// String removal - remove consecutive customers
inline CVRPSolution destroy_string(const CVRPSolution& sol, int n_remove,
                                   const CVRPInstance& inst, std::mt19937& rng) {
    CVRPSolution new_sol = sol.copy();
    
    // Collect all positions
    struct Position {
        int route_idx;
        int pos;
    };
    
    std::vector<Position> positions;
    for (size_t i = 0; i < new_sol.routes.size(); i++) {
        for (size_t j = 0; j < new_sol.routes[i].size(); j++) {
            positions.push_back({(int)i, (int)j});
        }
    }
    
    if (positions.empty()) return new_sol;
    
    // Select random starting position
    std::uniform_int_distribution<> dist(0, positions.size() - 1);
    Position start = positions[dist(rng)];
    
    // Remove consecutive customers from that route
    std::unordered_set<int> removed_set;
    int route_idx = start.route_idx;
    int pos = start.pos;
    int removed_count = 0;
    
    while (removed_count < n_remove && pos < (int)new_sol.routes[route_idx].size()) {
        removed_set.insert(new_sol.routes[route_idx][pos]);
        pos++;
        removed_count++;
    }
    
    // Remove from routes
    for (size_t i = 0; i < new_sol.routes.size(); i++) {
        auto& route = new_sol.routes[i];
        route.erase(
            std::remove_if(route.begin(), route.end(),
                          [&](int c) { return removed_set.count(c) > 0; }),
            route.end()
        );
        new_sol.update_route_load(i, inst);
    }
    
    new_sol.remove_empty_routes();
    return new_sol;
}

// Register all destroy operators
inline std::vector<std::pair<std::string, DestroyFunc>> get_destroy_operators() {
    return {
        {"random", destroy_random},
        {"worst", destroy_worst},
        {"shaw", destroy_shaw},
        {"string", destroy_string}
    };
}
// BLOCK A END


//==============================================================================
// BLOCK B START: REPAIR OPERATORS
//==============================================================================
// FM can modify any function in this block, or add new repair operators

// Greedy repair with O(1) capacity check
inline CVRPSolution repair_greedy(CVRPSolution& destroyed,
                                   const std::vector<int>& removed,
                                   const CVRPInstance& inst) {
    CVRPSolution sol = destroyed.copy();
    
    for (int customer : removed) {
        double best_cost = std::numeric_limits<double>::infinity();
        int best_route_idx = -1;
        int best_pos = -1;
        
        // Try inserting in existing routes
        for (size_t ri = 0; ri < sol.routes.size(); ri++) {
            // O(1) capacity check using cached loads
            if (!sol.can_insert(ri, customer, inst)) {
                continue;
            }
            
            const auto& route = sol.routes[ri];
            
            // Try all positions
            for (size_t pos = 0; pos <= route.size(); pos++) {
                int prev = (pos == 0) ? 0 : route[pos - 1];
                int next = (pos == route.size()) ? 0 : route[pos];
                
                // Cost increase
                double cost_before = inst.distance(prev, next);
                double cost_after = inst.distance(prev, customer) + inst.distance(customer, next);
                double cost_increase = cost_after - cost_before;
                
                if (cost_increase < best_cost) {
                    best_cost = cost_increase;
                    best_route_idx = ri;
                    best_pos = pos;
                }
            }
        }
        
        // Insert at best position or create new route
        if (best_route_idx != -1) {
            sol.routes[best_route_idx].insert(
                sol.routes[best_route_idx].begin() + best_pos,
                customer
            );
            // Update cached load - O(1) incremental update
            sol.route_loads[best_route_idx] += inst.demands[customer - 1];
        } else {
            // Create new route
            sol.routes.push_back({customer});
            sol.route_loads.push_back(inst.demands[customer - 1]);
        }
    }
    
    return sol;
}

// Regret-k repair
inline CVRPSolution repair_regret(CVRPSolution& destroyed,
                                   const std::vector<int>& removed,
                                   const CVRPInstance& inst) {
    CVRPSolution sol = destroyed.copy();
    std::vector<int> unrouted = removed;
    int k = 2;  // Regret-2
    
    struct InsertionOption {
        double cost;
        int route_idx;
        int pos;
        
        bool operator<(const InsertionOption& other) const {
            return cost < other.cost;
        }
    };
    
    while (!unrouted.empty()) {
        struct CustomerRegret {
            double regret;
            int customer;
            InsertionOption best_insertion;
        };
        
        std::vector<CustomerRegret> regrets;
        regrets.reserve(unrouted.size());
        
        for (int customer : unrouted) {
            std::vector<InsertionOption> options;
            options.reserve(sol.routes.size() * 10);
            
            // Try existing routes
            for (size_t ri = 0; ri < sol.routes.size(); ri++) {
                if (!sol.can_insert(ri, customer, inst)) {
                    continue;
                }
                
                const auto& route = sol.routes[ri];
                
                for (size_t pos = 0; pos <= route.size(); pos++) {
                    int prev = (pos == 0) ? 0 : route[pos - 1];
                    int next = (pos == route.size()) ? 0 : route[pos];
                    
                    double cost_before = inst.distance(prev, next);
                    double cost_after = inst.distance(prev, customer) + inst.distance(customer, next);
                    double cost_increase = cost_after - cost_before;
                    
                    options.push_back({cost_increase, (int)ri, (int)pos});
                }
            }
            
            // New route option
            double new_route_cost = 2.0 * inst.distance(0, customer);
            options.push_back({new_route_cost, -1, 0});
            
            if (options.empty()) continue;
            
            // Partial sort to get k best options
            int to_sort = std::min(k, (int)options.size());
            std::partial_sort(options.begin(), options.begin() + to_sort, options.end());
            
            // Calculate regret
            double best_cost = options[0].cost;
            double kth_cost = options[std::min(k - 1, (int)options.size() - 1)].cost;
            double regret_value = kth_cost - best_cost;
            
            regrets.push_back({regret_value, customer, options[0]});
        }
        
        if (regrets.empty()) break;
        
        // Find customer with max regret
        auto max_regret_it = std::max_element(
            regrets.begin(), regrets.end(),
            [](const CustomerRegret& a, const CustomerRegret& b) {
                return a.regret < b.regret;
            }
        );
        
        int customer = max_regret_it->customer;
        const auto& best = max_regret_it->best_insertion;
        
        // Insert customer
        if (best.route_idx == -1) {
            // Create new route
            sol.routes.push_back({customer});
            sol.route_loads.push_back(inst.demands[customer - 1]);
        } else {
            // Insert in existing route
            sol.routes[best.route_idx].insert(
                sol.routes[best.route_idx].begin() + best.pos,
                customer
            );
            sol.route_loads[best.route_idx] += inst.demands[customer - 1];
        }
        
        // Remove from unrouted
        unrouted.erase(std::remove(unrouted.begin(), unrouted.end(), customer), unrouted.end());
    }
    
    return sol;
}

// Register all repair operators
inline std::vector<std::pair<std::string, RepairFunc>> get_repair_operators() {
    return {
        {"greedy", repair_greedy},
        {"regret", repair_regret}
    };
}
// BLOCK B END


//==============================================================================
// BLOCK C START: INITIAL SOLUTION CONSTRUCTORS
//==============================================================================
// FM can modify any function in this block, or add new init methods

// Greedy insertion: insert customers one by one at best position
inline CVRPSolution init_greedy(const CVRPInstance& inst, std::mt19937& rng) {
    CVRPSolution sol;
    sol.routes.clear();
    
    // Create list of all customers
    std::vector<int> unrouted;
    for (int i = 1; i <= inst.n_customers; i++) {
        unrouted.push_back(i);
    }
    
    // Shuffle for randomness
    std::shuffle(unrouted.begin(), unrouted.end(), rng);
    
    // Insert each customer
    for (int customer : unrouted) {
        double best_cost = std::numeric_limits<double>::infinity();
        int best_route_idx = -1;
        int best_pos = -1;
        
        // Try inserting in existing routes
        for (size_t ri = 0; ri < sol.routes.size(); ri++) {
            const auto& route = sol.routes[ri];
            
            // Check capacity
            int load = 0;
            for (int c : route) load += inst.demands[c - 1];
            if (load + inst.demands[customer - 1] > inst.capacity) {
                continue;
            }
            
            // Try all positions
            for (size_t pos = 0; pos <= route.size(); pos++) {
                int prev = (pos == 0) ? 0 : route[pos - 1];
                int next = (pos == route.size()) ? 0 : route[pos];
                
                double cost_increase = -inst.distance(prev, next)
                                     + inst.distance(prev, customer)
                                     + inst.distance(customer, next);
                
                if (cost_increase < best_cost) {
                    best_cost = cost_increase;
                    best_route_idx = ri;
                    best_pos = pos;
                }
            }
        }
        
        // Also try creating new route
        double new_route_cost = 2.0 * inst.distance(0, customer);
        if (new_route_cost < best_cost) {
            best_route_idx = -1;
        }
        
        // Insert
        if (best_route_idx == -1) {
            // Create new route
            sol.routes.push_back({customer});
        } else {
            // Insert in existing route
            sol.routes[best_route_idx].insert(
                sol.routes[best_route_idx].begin() + best_pos,
                customer
            );
        }
    }
    
    // Evaluate and build loads
    sol.rebuild_route_loads(inst);
    evaluate_cvrp(sol, inst);
    
    return sol;
}

// Nearest neighbor heuristic
inline CVRPSolution init_nearest_neighbor(const CVRPInstance& inst, std::mt19937& rng) {
    CVRPSolution sol;
    std::vector<bool> visited(inst.n_customers + 1, false);
    visited[0] = true;  // depot
    
    int remaining = inst.n_customers;
    
    while (remaining > 0) {
        std::vector<int> route;
        int current_load = 0;
        int current = 0;  // Start from depot
        
        while (true) {
            // Find nearest unvisited customer that fits capacity
            int best = -1;
            double best_dist = std::numeric_limits<double>::infinity();
            
            for (int c = 1; c <= inst.n_customers; c++) {
                if (!visited[c]) {
                    int demand = inst.demands[c - 1];
                    if (current_load + demand <= inst.capacity) {
                        double d = inst.distance(current, c);
                        if (d < best_dist) {
                            best_dist = d;
                            best = c;
                        }
                    }
                }
            }
            
            if (best == -1) break;  // No more customers fit
            
            route.push_back(best);
            visited[best] = true;
            current = best;
            current_load += inst.demands[best - 1];
            remaining--;
        }
        
        if (!route.empty()) {
            sol.routes.push_back(route);
        }
    }
    
    sol.rebuild_route_loads(inst);
    evaluate_cvrp(sol, inst);
    
    return sol;
}

// Random insertion (baseline)
inline CVRPSolution init_random(const CVRPInstance& inst, std::mt19937& rng) {
    (void)rng;  // Unused
    CVRPSolution sol;
    
    // Create one route per customer
    for (int i = 1; i <= inst.n_customers; i++) {
        sol.routes.push_back({i});
    }
    
    sol.rebuild_route_loads(inst);
    evaluate_cvrp(sol, inst);
    
    return sol;
}

// Register all init methods
inline std::vector<std::pair<std::string, InitFunc>> get_init_methods() {
    return {
        {"greedy", init_greedy},
        {"nn", init_nearest_neighbor},
        {"random", init_random}
    };
}
// BLOCK C END


//==============================================================================
// BLOCK D START: LOCAL SEARCH OPERATORS
//==============================================================================
// FM can modify any function in this block, or add new local search operators

// Simple 2-opt for a single route (swap edges)
inline bool ls_two_opt_route(std::vector<int>& route, const CVRPInstance& inst) {
    int n = route.size();
    if (n < 2) return false;
    
    bool improved = false;
    
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            // Current edges: (prev[i] -> route[i]) and (route[j] -> next[j])
            // New edges: (prev[i] -> route[j]) and (route[i] -> next[j])
            
            int prev_i = (i == 0) ? 0 : route[i - 1];
            int next_j = (j == n - 1) ? 0 : route[j + 1];
            
            double old_cost = inst.distance(prev_i, route[i]) +
                            inst.distance(route[j], next_j);
            double new_cost = inst.distance(prev_i, route[j]) +
                            inst.distance(route[i], next_j);
            
            if (new_cost < old_cost - 1e-6) {
                // Reverse segment [i...j]
                std::reverse(route.begin() + i, route.begin() + j + 1);
                improved = true;
                break;
            }
        }
        if (improved) break;
    }
    
    return improved;
}

// Apply lightweight 2-opt to all routes
inline void ls_two_opt(CVRPSolution& sol, const CVRPInstance& inst) {
    for (int iter = 0; iter < 2; iter++) {
        bool any_improved = false;
        for (auto& route : sol.routes) {
            if (ls_two_opt_route(route, inst)) {
                any_improved = true;
            }
        }
        if (!any_improved) break;
    }
    
    evaluate_cvrp(sol, inst);
    sol.rebuild_route_loads(inst);
}

// Register all local search operators
inline std::vector<std::pair<std::string, LocalSearchFunc>> get_local_search_operators() {
    return {
        {"two_opt", ls_two_opt}
    };
}
// BLOCK D END
