#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ant.hpp"
#include "fractal_land.hpp"
#include "pheronome.hpp"
#include "rand_generator.hpp"
#include "renderer.hpp"
#include "window.hpp"

using steady_clock_t = std::chrono::steady_clock;

struct StepTiming {
    double ants_ms{0.0};
    double ant_select_move_ms{0.0};
    double ant_terrain_cost_ms{0.0};
    double ant_mark_pheromone_ms{0.0};
    std::size_t ant_moves{0};
    double evaporation_ms{0.0};
    double pheromone_update_ms{0.0};
    double advance_time_ms{0.0};
    double render_display_ms{0.0};
    double blit_ms{0.0};
    double event_poll_ms{0.0};
    double loop_ms{0.0};
};

struct RunTiming {
    double sdl_init_ms{0.0};
    double land_generation_ms{0.0};
    double land_normalization_ms{0.0};
    double ant_init_ms{0.0};
    double pheromone_init_ms{0.0};
    double window_init_ms{0.0};
    double renderer_init_ms{0.0};
    double setup_total_ms{0.0};
    StepTiming loop_steps{};
    std::size_t nb_ants{0};
    std::size_t grid_dim{0};
    std::size_t iterations{0};
    std::size_t delivered_food{0};
    double run_total_ms{0.0};
};

struct ProgramOptions {
    bool benchmark{false};
    std::size_t runs{5};
    std::size_t iterations{1000};
    std::size_t ants{5000};
    bool render{false};
    std::string csv_path{"../output/timing_results.csv"};
};

static double to_ms(steady_clock_t::time_point t0, steady_clock_t::time_point t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double safe_div(double numerator, double denominator)
{
    return (denominator > 0.0) ? (numerator / denominator) : 0.0;
}

void advance_time(const fractal_land& land, pheronome& phen, const position_t& pos_nest, const position_t& pos_food,
                  std::vector<ant>& ants, std::size_t& cpteur, StepTiming* timing = nullptr)
{
    auto t0 = steady_clock_t::now();

    auto ants_t0 = steady_clock_t::now();
    ant::step_timing ant_details;
    for (std::size_t i = 0; i < ants.size(); ++i)
        ants[i].advance(phen, land, pos_food, pos_nest, cpteur, &ant_details);
    auto ants_t1 = steady_clock_t::now();

    auto evap_t0 = steady_clock_t::now();
    phen.do_evaporation();
    auto evap_t1 = steady_clock_t::now();

    auto upd_t0 = steady_clock_t::now();
    phen.update();
    auto upd_t1 = steady_clock_t::now();

    if (timing != nullptr) {
        timing->ants_ms += to_ms(ants_t0, ants_t1);
        timing->ant_select_move_ms += ant_details.select_move_ms;
        timing->ant_terrain_cost_ms += ant_details.terrain_cost_ms;
        timing->ant_mark_pheromone_ms += ant_details.mark_pheromone_ms;
        timing->ant_moves += ant_details.moves;
        timing->evaporation_ms += to_ms(evap_t0, evap_t1);
        timing->pheromone_update_ms += to_ms(upd_t0, upd_t1);
        timing->advance_time_ms += to_ms(t0, steady_clock_t::now());
    }
}

static void print_usage(const char* binary_name)
{
    std::cout << "Usage:\n";
    std::cout << "  " << binary_name << "                       # interactive mode (render enabled)\n";
    std::cout << "  " << binary_name << " --benchmark [options]\n";
    std::cout << "Options for benchmark:\n";
    std::cout << "  --runs <N>          number of runs (default: 5)\n";
    std::cout << "  --iterations <N>    number of iterations per run (default: 1000)\n";
    std::cout << "  --ants <N>          number of ants (default: 5000)\n";
    std::cout << "  --render <0|1>      include rendering in benchmark (default: 0)\n";
    std::cout << "  --csv <path>        csv output path (default: ../output/timing_results.csv)\n";
}

static bool parse_size_t_arg(const char* arg, std::size_t& out_val)
{
    try {
        out_val = static_cast<std::size_t>(std::stoull(arg));
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_options(int argc, char* argv[], ProgramOptions& opt)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--benchmark") {
            opt.benchmark = true;
            continue;
        }
        if (arg == "--runs" && i + 1 < argc) {
            if (!parse_size_t_arg(argv[++i], opt.runs)) return false;
            continue;
        }
        if (arg == "--iterations" && i + 1 < argc) {
            if (!parse_size_t_arg(argv[++i], opt.iterations)) return false;
            continue;
        }
        if (arg == "--ants" && i + 1 < argc) {
            if (!parse_size_t_arg(argv[++i], opt.ants)) return false;
            continue;
        }
        if (arg == "--csv" && i + 1 < argc) {
            opt.csv_path = argv[++i];
            continue;
        }
        if (arg == "--render" && i + 1 < argc) {
            std::size_t render_flag = 0;
            if (!parse_size_t_arg(argv[++i], render_flag) || render_flag > 1) return false;
            opt.render = (render_flag == 1);
            continue;
        }
        return false;
    }
    return true;
}

static RunTiming run_once(bool interactive_mode, std::size_t max_iterations, bool enable_render, std::size_t nb_ants)
{
    RunTiming stats;
    auto run_t0 = steady_clock_t::now();

    auto sdl_t0 = steady_clock_t::now();
    Uint32 sdl_flags = enable_render ? SDL_INIT_VIDEO : SDL_INIT_TIMER;
    if (SDL_Init(sdl_flags) != 0) {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        return stats;
    }
    auto sdl_t1 = steady_clock_t::now();
    stats.sdl_init_ms = to_ms(sdl_t0, sdl_t1);

    auto setup_t0 = steady_clock_t::now();

    std::size_t seed = 2026;
    const double eps = 0.8;
    const double alpha = 0.7;
    const double beta = 0.999;
    position_t pos_nest{256, 256};
    position_t pos_food{500, 500};

    auto land_t0 = steady_clock_t::now();
    fractal_land land(8, 2, 1.0, 1024);
    auto land_t1 = steady_clock_t::now();
    stats.land_generation_ms = to_ms(land_t0, land_t1);
    stats.nb_ants = nb_ants;
    stats.grid_dim = land.dimensions();

    auto norm_t0 = steady_clock_t::now();
    double max_val = 0.0;
    double min_val = 0.0;
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
            max_val = std::max(max_val, land(i, j));
            min_val = std::min(min_val, land(i, j));
        }
    double delta = max_val - min_val;
    for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
        for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j)
            land(i, j) = (land(i, j) - min_val) / delta;
    auto norm_t1 = steady_clock_t::now();
    stats.land_normalization_ms = to_ms(norm_t0, norm_t1);

    auto ants_init_t0 = steady_clock_t::now();
    ant::set_exploration_coef(eps);
    std::vector<ant> ants;
    ants.reserve(nb_ants);
    auto gen_ant_pos = [&land, &seed]() { return rand_int32(0, land.dimensions() - 1, seed); };
    for (std::size_t i = 0; i < nb_ants; ++i)
        ants.emplace_back(position_t{gen_ant_pos(), gen_ant_pos()}, seed);
    auto ants_init_t1 = steady_clock_t::now();
    stats.ant_init_ms = to_ms(ants_init_t0, ants_init_t1);

    auto phen_t0 = steady_clock_t::now();
    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);
    auto phen_t1 = steady_clock_t::now();
    stats.pheromone_init_ms = to_ms(phen_t0, phen_t1);

    std::unique_ptr<Window> win;
    std::unique_ptr<Renderer> renderer;
    if (enable_render) {
        auto win_t0 = steady_clock_t::now();
        win = std::make_unique<Window>("Ant Simulation", 2 * static_cast<int>(land.dimensions()) + 10,
                                       static_cast<int>(land.dimensions()) + 266);
        auto win_t1 = steady_clock_t::now();
        stats.window_init_ms = to_ms(win_t0, win_t1);

        auto ren_t0 = steady_clock_t::now();
        renderer = std::make_unique<Renderer>(land, phen, pos_nest, pos_food, ants);
        auto ren_t1 = steady_clock_t::now();
        stats.renderer_init_ms = to_ms(ren_t0, ren_t1);
    }

    auto setup_t1 = steady_clock_t::now();
    stats.setup_total_ms = to_ms(setup_t0, setup_t1);

    std::size_t food_quantity = 0;
    SDL_Event event;
    bool cont_loop = true;
    bool not_food_in_nest = true;
    std::size_t it = 0;

    while (cont_loop) {
        auto loop_t0 = steady_clock_t::now();
        ++it;

        if (interactive_mode) {
            auto poll_t0 = steady_clock_t::now();
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT)
                    cont_loop = false;
            }
            auto poll_t1 = steady_clock_t::now();
            stats.loop_steps.event_poll_ms += to_ms(poll_t0, poll_t1);
        }

        advance_time(land, phen, pos_nest, pos_food, ants, food_quantity, &stats.loop_steps);

        if (enable_render && renderer && win) {
            auto render_t0 = steady_clock_t::now();
            renderer->display(*win, food_quantity);
            auto render_t1 = steady_clock_t::now();
            stats.loop_steps.render_display_ms += to_ms(render_t0, render_t1);

            auto blit_t0 = steady_clock_t::now();
            win->blit();
            auto blit_t1 = steady_clock_t::now();
            stats.loop_steps.blit_ms += to_ms(blit_t0, blit_t1);
        }

        if (not_food_in_nest && food_quantity > 0) {
            std::cout << "La premiere nourriture est arrivee au nid a l'iteration " << it << std::endl;
            not_food_in_nest = false;
        }

        auto loop_t1 = steady_clock_t::now();
        stats.loop_steps.loop_ms += to_ms(loop_t0, loop_t1);
        stats.iterations = it;
        stats.delivered_food = food_quantity;

        if (!interactive_mode && it >= max_iterations)
            cont_loop = false;
    }

    SDL_Quit();
    stats.run_total_ms = to_ms(run_t0, steady_clock_t::now());
    return stats;
}

static void write_csv(const std::string& path, const std::vector<RunTiming>& runs)
{
    std::filesystem::path out_path(path);
    if (out_path.has_parent_path()) {
        std::filesystem::create_directories(out_path.parent_path());
    }

    std::ofstream csv(path);
    if (!csv.is_open()) {
        std::cerr << "Failed to open csv file: " << path << std::endl;
        return;
    }

    csv << "run,nb_ants,grid_dim,iterations,ant_moves,work_ant_calls,work_grid_cells,sdl_init_ms,setup_total_ms,"
           "land_generation_ms,land_normalization_ms,ant_init_ms,pheromone_init_ms,window_init_ms,renderer_init_ms,"
           "event_poll_ms,advance_time_ms,ants_ms,ant_select_move_ms,ant_terrain_cost_ms,ant_mark_pheromone_ms,"
           "evaporation_ms,pheromone_update_ms,render_display_ms,blit_ms,loop_ms,delivered_food,run_total_ms,"
           "ants_us_per_ant_call,ant_select_ns_per_move,ant_terrain_ns_per_move,ant_mark_ns_per_move,"
           "evaporation_ns_per_cell,update_us_per_iter\n";

    RunTiming avg;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const RunTiming& r = runs[i];
        const double ant_calls = static_cast<double>(r.nb_ants) * static_cast<double>(r.iterations);
        const double grid_cells = static_cast<double>(r.grid_dim) * static_cast<double>(r.grid_dim) * static_cast<double>(r.iterations);
        const double ant_moves = static_cast<double>(r.loop_steps.ant_moves);
        csv << (i + 1) << "," << r.nb_ants << "," << r.grid_dim << "," << r.iterations << "," << r.loop_steps.ant_moves << ","
            << ant_calls << "," << grid_cells << "," << r.sdl_init_ms << "," << r.setup_total_ms << ","
            << r.land_generation_ms << "," << r.land_normalization_ms << "," << r.ant_init_ms << ","
            << r.pheromone_init_ms << "," << r.window_init_ms << "," << r.renderer_init_ms << ","
            << r.loop_steps.event_poll_ms << "," << r.loop_steps.advance_time_ms << "," << r.loop_steps.ants_ms << ","
            << r.loop_steps.ant_select_move_ms << "," << r.loop_steps.ant_terrain_cost_ms << ","
            << r.loop_steps.ant_mark_pheromone_ms << "," << r.loop_steps.evaporation_ms << ","
            << r.loop_steps.pheromone_update_ms << "," << r.loop_steps.render_display_ms << "," << r.loop_steps.blit_ms
            << "," << r.loop_steps.loop_ms << "," << r.delivered_food << "," << r.run_total_ms << ","
            << safe_div(r.loop_steps.ants_ms * 1000.0, ant_calls) << ","
            << safe_div(r.loop_steps.ant_select_move_ms * 1e6, ant_moves) << ","
            << safe_div(r.loop_steps.ant_terrain_cost_ms * 1e6, ant_moves) << ","
            << safe_div(r.loop_steps.ant_mark_pheromone_ms * 1e6, ant_moves) << ","
            << safe_div(r.loop_steps.evaporation_ms * 1e6, grid_cells) << ","
            << safe_div(r.loop_steps.pheromone_update_ms * 1000.0, static_cast<double>(r.iterations)) << "\n";

        avg.sdl_init_ms += r.sdl_init_ms;
        avg.setup_total_ms += r.setup_total_ms;
        avg.land_generation_ms += r.land_generation_ms;
        avg.land_normalization_ms += r.land_normalization_ms;
        avg.ant_init_ms += r.ant_init_ms;
        avg.pheromone_init_ms += r.pheromone_init_ms;
        avg.window_init_ms += r.window_init_ms;
        avg.renderer_init_ms += r.renderer_init_ms;
        avg.nb_ants += r.nb_ants;
        avg.grid_dim += r.grid_dim;
        avg.iterations += r.iterations;
        avg.loop_steps.event_poll_ms += r.loop_steps.event_poll_ms;
        avg.loop_steps.advance_time_ms += r.loop_steps.advance_time_ms;
        avg.loop_steps.ants_ms += r.loop_steps.ants_ms;
        avg.loop_steps.ant_select_move_ms += r.loop_steps.ant_select_move_ms;
        avg.loop_steps.ant_terrain_cost_ms += r.loop_steps.ant_terrain_cost_ms;
        avg.loop_steps.ant_mark_pheromone_ms += r.loop_steps.ant_mark_pheromone_ms;
        avg.loop_steps.ant_moves += r.loop_steps.ant_moves;
        avg.loop_steps.evaporation_ms += r.loop_steps.evaporation_ms;
        avg.loop_steps.pheromone_update_ms += r.loop_steps.pheromone_update_ms;
        avg.loop_steps.render_display_ms += r.loop_steps.render_display_ms;
        avg.loop_steps.blit_ms += r.loop_steps.blit_ms;
        avg.loop_steps.loop_ms += r.loop_steps.loop_ms;
        avg.delivered_food += r.delivered_food;
        avg.run_total_ms += r.run_total_ms;
    }

    if (!runs.empty()) {
        const double inv_n = 1.0 / static_cast<double>(runs.size());
        const double avg_ants = static_cast<double>(avg.nb_ants) * inv_n;
        const double avg_dim = static_cast<double>(avg.grid_dim) * inv_n;
        const double avg_iter = static_cast<double>(avg.iterations) * inv_n;
        const double avg_moves = static_cast<double>(avg.loop_steps.ant_moves) * inv_n;
        const double avg_ant_calls = avg_ants * avg_iter;
        const double avg_grid_cells = avg_dim * avg_dim * avg_iter;
        csv << "avg" << "," << avg_ants << "," << avg_dim << "," << avg_iter << "," << avg_moves << ","
            << avg_ant_calls << "," << avg_grid_cells << "," << avg.sdl_init_ms * inv_n << "," << avg.setup_total_ms * inv_n << ","
            << avg.land_generation_ms * inv_n << "," << avg.land_normalization_ms * inv_n << "," << avg.ant_init_ms * inv_n
            << "," << avg.pheromone_init_ms * inv_n << "," << avg.window_init_ms * inv_n << ","
            << avg.renderer_init_ms * inv_n << "," << avg.loop_steps.event_poll_ms * inv_n << ","
            << avg.loop_steps.advance_time_ms * inv_n << "," << avg.loop_steps.ants_ms * inv_n << ","
            << avg.loop_steps.ant_select_move_ms * inv_n << "," << avg.loop_steps.ant_terrain_cost_ms * inv_n << ","
            << avg.loop_steps.ant_mark_pheromone_ms * inv_n << "," << avg.loop_steps.evaporation_ms * inv_n << ","
            << avg.loop_steps.pheromone_update_ms * inv_n << "," << avg.loop_steps.render_display_ms * inv_n << ","
            << avg.loop_steps.blit_ms * inv_n << "," << avg.loop_steps.loop_ms * inv_n << ","
            << static_cast<double>(avg.delivered_food) * inv_n << "," << avg.run_total_ms * inv_n << ","
            << safe_div((avg.loop_steps.ants_ms * inv_n) * 1000.0, avg_ant_calls) << ","
            << safe_div((avg.loop_steps.ant_select_move_ms * inv_n) * 1e6, avg_moves) << ","
            << safe_div((avg.loop_steps.ant_terrain_cost_ms * inv_n) * 1e6, avg_moves) << ","
            << safe_div((avg.loop_steps.ant_mark_pheromone_ms * inv_n) * 1e6, avg_moves) << ","
            << safe_div((avg.loop_steps.evaporation_ms * inv_n) * 1e6, avg_grid_cells) << ","
            << safe_div((avg.loop_steps.pheromone_update_ms * inv_n) * 1000.0, avg_iter) << "\n";
    }
}

int main(int argc, char* argv[])
{
    ProgramOptions opt;
    if (!parse_options(argc, argv, opt)) {
        print_usage(argv[0]);
        return 1;
    }

    if (!opt.benchmark) {
        run_once(true, 0, true, opt.ants);
        return 0;
    }

    std::vector<RunTiming> runs;
    runs.reserve(opt.runs);
    for (std::size_t i = 0; i < opt.runs; ++i) {
        std::cout << "[benchmark] run " << (i + 1) << "/" << opt.runs << std::endl;
        runs.emplace_back(run_once(false, opt.iterations, opt.render, opt.ants));
    }

    write_csv(opt.csv_path, runs);
    std::cout << "Timing CSV written to: " << opt.csv_path << std::endl;
    return 0;
}
