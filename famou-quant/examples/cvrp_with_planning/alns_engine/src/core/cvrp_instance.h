#pragma once

#include <vector>
#include <cmath>
#include <string>

struct Point {
    double x, y;
    Point() : x(0), y(0) {}
    Point(double x_, double y_) : x(x_), y(y_) {}
};

class CVRPInstance {
public:
    Point depot;
    std::vector<Point> customers;
    std::vector<int> demands;
    int capacity;
    int n_customers;

    // Pre-computed distance matrix (flat array for cache efficiency)
    std::vector<double> dist_matrix;

    CVRPInstance() : capacity(0), n_customers(0) {}

    CVRPInstance(const Point& dep, const std::vector<Point>& custs,
                 const std::vector<int>& dems, int cap)
        : depot(dep), customers(custs), demands(dems), capacity(cap) {
        n_customers = customers.size();
        compute_distances();
    }

    // Inline distance lookup - O(1)
    inline double distance(int i, int j) const {
        int n = n_customers + 1;
        return dist_matrix[i * n + j];
    }

private:
    void compute_distances() {
        int n = n_customers + 1;  // depot + customers
        dist_matrix.resize(n * n);

        // All locations (depot first)
        std::vector<Point> locations;
        locations.reserve(n);
        locations.push_back(depot);
        locations.insert(locations.end(), customers.begin(), customers.end());

        // Compute symmetric distance matrix
        for (int i = 0; i < n; i++) {
            dist_matrix[i * n + i] = 0.0;
            for (int j = i + 1; j < n; j++) {
                double dx = locations[i].x - locations[j].x;
                double dy = locations[i].y - locations[j].y;
                double d = std::sqrt(dx * dx + dy * dy);
                dist_matrix[i * n + j] = d;
                dist_matrix[j * n + i] = d;
            }
        }
    }
};

class CVRPSolution {
public:
    std::vector<std::vector<int>> routes;
    double cost;

    // Cached route loads for O(1) capacity check
    std::vector<int> route_loads;

    CVRPSolution() : cost(-1.0) {}

    explicit CVRPSolution(const std::vector<std::vector<int>>& r)
        : routes(r), cost(-1.0) {
        route_loads.resize(routes.size(), 0);
    }

    // Efficient shallow copy with vector copy constructor
    CVRPSolution copy() const {
        CVRPSolution sol;
        sol.routes = routes;  // vector copy constructor
        sol.cost = cost;
        sol.route_loads = route_loads;
        return sol;
    }

    // Update route load cache - O(1) update
    void update_route_load(int route_idx, const CVRPInstance& inst) {
        int load = 0;
        for (int c : routes[route_idx]) {
            load += inst.demands[c - 1];
        }

        // Ensure route_loads has correct size
        if (route_idx >= (int)route_loads.size()) {
            route_loads.resize(route_idx + 1, 0);
        }
        route_loads[route_idx] = load;
    }

    // Rebuild all route load cache
    void rebuild_route_loads(const CVRPInstance& inst) {
        route_loads.resize(routes.size());
        for (size_t i = 0; i < routes.size(); i++) {
            update_route_load(i, inst);
        }
    }

    // O(1) capacity check using cache
    inline bool can_insert(int route_idx, int customer, const CVRPInstance& inst) const {
        if (route_idx >= (int)route_loads.size()) return false;
        return route_loads[route_idx] + inst.demands[customer - 1] <= inst.capacity;
    }

    // Remove empty routes
    void remove_empty_routes() {
        auto it = routes.begin();
        auto load_it = route_loads.begin();
        while (it != routes.end()) {
            if (it->empty()) {
                it = routes.erase(it);
                load_it = route_loads.erase(load_it);
            } else {
                ++it;
                ++load_it;
            }
        }
    }
};
