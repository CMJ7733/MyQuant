#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <string>
#include <queue>

#include "core/cvrp_instance.h"
#include "core/cvrp_evaluate.h"
#include "core/data_loader.h"
#include "core/alns_framework.h"
#include "alns_operators.h"

struct InstanceResult {
    int instance_idx;
    double initial_cost;
    double final_cost;
    double improvement;
    double improvement_pct;
    int n_routes;
    bool feasible;
    int iters;
    double time_s;
    int seed;
};

// Global data and synchronization
std::vector<CVRPInstance> g_instances;
std::queue<int> g_task_queue;
std::mutex g_queue_mutex;
std::vector<InstanceResult> g_results;
std::mutex g_results_mutex;
std::atomic<int> g_completed{0};
int g_total = 0;
std::mutex g_output_mutex;
auto g_start_time = std::chrono::steady_clock::now();

void print_progress() {
    int done = g_completed.load();
    double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - g_start_time
    ).count();

    double rate = elapsed > 0 ? done / elapsed : 0.0;
    double eta = rate > 0 ? (g_total - done) / rate : 0.0;

    int bar_len = 30;
    int filled = g_total > 0 ? bar_len * done / g_total : bar_len;
    std::string bar(filled, '=');
    bar += std::string(bar_len - filled, '-');

    std::cout << "\r[" << bar << "] "
              << done << "/" << g_total
              << " (" << std::fixed << std::setprecision(1)
              << (g_total > 0 ? 100.0 * done / g_total : 100.0) << "%) "
              << "ETA " << std::setprecision(1) << eta << "s    " << std::flush;
}

InstanceResult solve_instance(
    int idx,
    double time_limit,
    int seed
) {
    const auto& inst = g_instances[idx];

    // Generate initial solution (from BLOCK C)
    std::mt19937 init_rng(seed + idx);
    auto init_methods = get_init_methods();
    CVRPSolution initial = init_methods[0].second(inst, init_rng);  // Use first method (greedy)
    double initial_cost = initial.cost;

    // Get operators from BLOCK A and B
    auto destroy_ops = get_destroy_operators();
    auto repair_ops = get_repair_operators();
    auto ls_ops = get_local_search_operators();

    // Setup ALNS Framework
    ALNSFramework alns(
        inst,
        destroy_ops,
        repair_ops,
        ls_ops.empty() ? nullptr : ls_ops[0].second,  // Use first LS if available
        500.0,   // T_start
        0.01,    // T_end
        0.9995,  // cooling_rate
        100,     // segment_size
        seed + idx,
        false    // verbose
    );

    // Run
    auto start = std::chrono::steady_clock::now();
    CVRPSolution best = alns.solve(initial, time_limit);
    auto end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(end - start).count();

    // Result
    InstanceResult result;
    result.instance_idx = idx;
    result.initial_cost = initial_cost;
    result.final_cost = best.cost;
    result.improvement = initial_cost - best.cost;
    result.improvement_pct = (initial_cost - best.cost) / initial_cost * 100.0;
    result.n_routes = best.routes.size();
    result.feasible = is_feasible(best, inst);
    result.iters = alns.iters;
    result.time_s = elapsed;
    result.seed = seed + idx;

    return result;
}

void worker_thread(
    double time_limit,
    int seed
) {
    while (true) {
        int idx = -1;

        // Get next instance from queue
        {
            std::lock_guard<std::mutex> lock(g_queue_mutex);
            if (g_task_queue.empty()) break;
            idx = g_task_queue.front();
            g_task_queue.pop();
        }

        // Solve
        try {
            InstanceResult result = solve_instance(idx, time_limit, seed);

            // Store result
            {
                std::lock_guard<std::mutex> lock(g_results_mutex);
                g_results.push_back(result);
            }

            // Update progress
            g_completed++;

            {
                std::lock_guard<std::mutex> lock(g_output_mutex);
                print_progress();
            }

        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(g_output_mutex);
            std::cout << "\nInstance " << idx << " failed: " << e.what() << std::endl;
        }
    }
}

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "Options:\n"
              << "  --data <path>     Path to data file (required)\n"
              << "  --time <float>    Time limit per instance in seconds (default: 60.0)\n"
              << "  --workers <int>   Number of parallel workers (default: 10)\n"
              << "  --seed <int>      Base random seed (default: 42)\n"
              << "  --start <int>     Start instance index (default: 0)\n"
              << "  --count <int>     Number of instances to evaluate (default: all)\n"
              << "  --output <path>   Output CSV file (optional)\n"
              << "  --help            Show this help\n";
}

int main(int argc, char** argv) {
    // Parse arguments
    std::string data_path;
    double time_limit = 60.0;
    int num_workers = 10;
    int seed = 42;
    int start_idx = 0;
    int count = -1;
    std::string output_path;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--data" && i + 1 < argc) {
            data_path = argv[++i];
        } else if (arg == "--time" && i + 1 < argc) {
            time_limit = std::stod(argv[++i]);
        } else if (arg == "--workers" && i + 1 < argc) {
            num_workers = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::stoi(argv[++i]);
        } else if (arg == "--start" && i + 1 < argc) {
            start_idx = std::stoi(argv[++i]);
        } else if (arg == "--count" && i + 1 < argc) {
            count = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (data_path.empty()) {
        std::cerr << "Error: --data is required\n";
        print_usage(argv[0]);
        return 1;
    }

    try {
        // Load data
        std::cout << "Loading data from " << data_path << "...\n";
        g_instances = DataLoader::load_txt(data_path);

        // Determine instances to evaluate
        int total_instances = g_instances.size();
        int end_idx = (count < 0) ? total_instances : std::min(total_instances, start_idx + count);

        for (int i = start_idx; i < end_idx; i++) {
            g_task_queue.push(i);
        }

        if (g_task_queue.empty()) {
            std::cout << "No instances to evaluate\n";
            return 1;
        }

        g_total = g_task_queue.size();

        std::cout << "Evaluating " << g_total << " instances\n";
        std::cout << "Time limit: " << time_limit << "s per instance\n";
        std::cout << "Workers: " << num_workers << "\n\n";

        // Start workers
        g_start_time = std::chrono::steady_clock::now();
        g_completed = 0;

        std::vector<std::thread> threads;
        for (int i = 0; i < num_workers; i++) {
            threads.emplace_back(worker_thread, time_limit, seed);
        }

        // Wait for completion
        for (auto& t : threads) {
            t.join();
        }

        std::cout << "\n\n";

        // Sort results by instance index
        std::sort(g_results.begin(), g_results.end(),
                 [](const InstanceResult& a, const InstanceResult& b) {
                     return a.instance_idx < b.instance_idx;
                 });

        // Save to CSV if requested
        if (!output_path.empty()) {
            std::ofstream csv(output_path);
            csv << "instance_idx,initial_cost,final_cost,improvement,improvement_pct,"
                << "n_routes,feasible,iters,time_s,seed\n";

            for (const auto& r : g_results) {
                csv << r.instance_idx << ","
                    << std::fixed << std::setprecision(4) << r.initial_cost << ","
                    << r.final_cost << ","
                    << r.improvement << ","
                    << std::setprecision(2) << r.improvement_pct << ","
                    << r.n_routes << ","
                    << (r.feasible ? 1 : 0) << ","
                    << r.iters << ","
                    << std::setprecision(2) << r.time_s << ","
                    << r.seed << "\n";
            }

            std::cout << "Results saved to " << output_path << "\n\n";
        }

        // Summary statistics
        double sum_initial = 0, sum_final = 0, sum_improvement = 0;
        int sum_iters = 0;
        double sum_time = 0;
        int feasible_count = 0;

        for (const auto& r : g_results) {
            sum_initial += r.initial_cost;
            sum_final += r.final_cost;
            sum_improvement += r.improvement_pct;
            sum_iters += r.iters;
            sum_time += r.time_s;
            if (r.feasible) feasible_count++;
        }

        int n = g_results.size();

        std::cout << "======================================================================\n";
        std::cout << "SUMMARY\n";
        std::cout << "======================================================================\n";
        std::cout << "Instances evaluated: " << n << "\n";
        std::cout << "Feasible rate: " << feasible_count << "/" << n
                  << " (" << std::fixed << std::setprecision(1)
                  << (100.0 * feasible_count / n) << "%)\n\n";
        std::cout << "Initial cost (avg): " << std::setprecision(4)
                  << (sum_initial / n) << "\n";
        std::cout << "Final cost (avg):   " << (sum_final / n) << "\n";
        std::cout << "Improvement (avg):  " << std::setprecision(2)
                  << (sum_improvement / n) << "%\n\n";
        std::cout << "Iterations (avg):   " << std::setprecision(1)
                  << (sum_iters / (double)n) << "\n";
        std::cout << "Time (avg):         " << (sum_time / n) << "s\n";
        std::cout << "Speed (avg):        " << std::setprecision(2)
                  << (sum_iters / sum_time) << " iter/s\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
