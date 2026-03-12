#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_MPI
#include <mpi.h>
#endif

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
    double mpi_mark_sync_ms{0.0};
    double mpi_evap_sync_ms{0.0};
    double submap_reassign_ms{0.0};
    std::size_t submap_border_crossings{0};
    std::size_t submap_border_marks_exchanged{0};
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
    std::size_t threads{1};
    bool parallel_enabled{false};
    bool vectorized_enabled{false};
    bool simd_enabled{false};
    bool submap_enabled{false};
    bool mpi_enabled{false};
    std::size_t mpi_processes{1};
    double run_total_ms{0.0};
};

struct ProgramOptions {
    bool benchmark{false};
    std::size_t runs{5};
    std::size_t iterations{1000};
    std::size_t ants{5000};
    bool parallel{true};
    bool vectorized{false};
    bool simd{false};
    bool submap{false};
    bool mpi{false};
    std::size_t threads{0};
    bool render{false};
    std::string csv_path{"../output/timing_results.csv"};
};

struct MpiContext {
    bool enabled{false};
    int rank{0};
    int size{1};
};

struct AntSoA {
    static constexpr std::uint8_t unloaded = 0;
    static constexpr std::uint8_t loaded = 1;

    std::vector<int> x;
    std::vector<int> y;
    std::vector<std::uint8_t> state;
    std::vector<std::size_t> seed;

    std::size_t size() const { return x.size(); }
    void reserve(std::size_t n)
    {
        x.reserve(n);
        y.reserve(n);
        state.reserve(n);
        seed.reserve(n);
    }
    void push_back(const position_t& pos, std::size_t ant_seed)
    {
        x.push_back(pos.x);
        y.push_back(pos.y);
        state.push_back(unloaded);
        seed.push_back(ant_seed);
    }
};

static double to_ms(steady_clock_t::time_point t0, steady_clock_t::time_point t1)
{
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

static double safe_div(double numerator, double denominator)
{
    return (denominator > 0.0) ? (numerator / denominator) : 0.0;
}

static void split_even_work(std::size_t total, std::size_t rank, std::size_t ranks, std::size_t& begin, std::size_t& end)
{
    const std::size_t base = total / ranks;
    const std::size_t rem = total % ranks;
    begin = rank * base + std::min(rank, rem);
    end = begin + base + (rank < rem ? 1 : 0);
}

static bool position_less(const position_t& lhs, const position_t& rhs)
{
    return (lhs.x < rhs.x) || ((lhs.x == rhs.x) && (lhs.y < rhs.y));
}

static bool position_equal(const position_t& lhs, const position_t& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

static std::size_t owner_from_row(int x, std::size_t dim, std::size_t owners)
{
    if (owners <= 1 || dim == 0) return 0;
    int clamped_x = std::max(0, std::min(x, static_cast<int>(dim) - 1));
    return (static_cast<std::size_t>(clamped_x) * owners) / dim;
}

void advance_time(const fractal_land& land, pheronome& phen, const position_t& pos_nest, const position_t& pos_food,
                  std::vector<ant>& ants, std::size_t& cpteur, bool parallel, bool simd, StepTiming* timing = nullptr)
{
    auto t0 = steady_clock_t::now();

    auto ants_t0 = steady_clock_t::now();
    ant::step_timing ant_details;
    std::vector<position_t> marked_cells;
    marked_cells.reserve(ants.size() * 2);

    if (!parallel) {
        for (std::size_t i = 0; i < ants.size(); ++i)
            ants[i].advance(phen, land, pos_food, pos_nest, cpteur, &ant_details, &marked_cells);
    } else {
#ifdef _OPENMP
        const int thread_count = omp_get_max_threads();
        std::vector<std::vector<position_t>> marked_per_thread(static_cast<std::size_t>(thread_count));
        std::vector<std::size_t> food_per_thread(static_cast<std::size_t>(thread_count), 0);
        std::vector<ant::step_timing> timing_per_thread(static_cast<std::size_t>(thread_count));

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& local_marked = marked_per_thread[static_cast<std::size_t>(tid)];
            auto& local_timing = timing_per_thread[static_cast<std::size_t>(tid)];
            std::size_t local_food = 0;
            local_marked.reserve((ants.size() / static_cast<std::size_t>(thread_count)) * 3 + 128);

#pragma omp for schedule(static)
            for (std::size_t i = 0; i < ants.size(); ++i) {
                ants[i].advance(phen, land, pos_food, pos_nest, local_food, &local_timing, &local_marked);
            }
            food_per_thread[static_cast<std::size_t>(tid)] = local_food;
        }

        for (std::size_t t = 0; t < marked_per_thread.size(); ++t) {
            cpteur += food_per_thread[t];
            ant_details.select_move_ms += timing_per_thread[t].select_move_ms;
            ant_details.terrain_cost_ms += timing_per_thread[t].terrain_cost_ms;
            ant_details.moves += timing_per_thread[t].moves;
            auto& v = marked_per_thread[t];
            marked_cells.insert(marked_cells.end(), v.begin(), v.end());
        }
#else
        for (std::size_t i = 0; i < ants.size(); ++i)
            ants[i].advance(phen, land, pos_food, pos_nest, cpteur, &ant_details, &marked_cells);
#endif
    }
    auto ants_t1 = steady_clock_t::now();

    auto mark_t0 = steady_clock_t::now();
    std::sort(marked_cells.begin(), marked_cells.end(), position_less);
    marked_cells.erase(std::unique(marked_cells.begin(), marked_cells.end(), position_equal), marked_cells.end());
    for (const auto& pos : marked_cells) {
        phen.mark_pheronome(pos);
    }
    auto mark_t1 = steady_clock_t::now();

    auto evap_t0 = steady_clock_t::now();
    phen.do_evaporation(simd);
    auto evap_t1 = steady_clock_t::now();

    auto upd_t0 = steady_clock_t::now();
    phen.update(simd);
    auto upd_t1 = steady_clock_t::now();

    if (timing != nullptr) {
        timing->ants_ms += to_ms(ants_t0, ants_t1);
        timing->ant_select_move_ms += ant_details.select_move_ms;
        timing->ant_terrain_cost_ms += ant_details.terrain_cost_ms;
        timing->ant_mark_pheromone_ms += to_ms(mark_t0, mark_t1);
        timing->ant_moves += ant_details.moves;
        timing->evaporation_ms += to_ms(evap_t0, evap_t1);
        timing->pheromone_update_ms += to_ms(upd_t0, upd_t1);
        timing->advance_time_ms += to_ms(t0, steady_clock_t::now());
    }
}

static void advance_time_mpi(const fractal_land& land, pheronome& phen, const position_t& pos_nest, const position_t& pos_food,
                             std::vector<ant>& ants, std::size_t& cpteur, bool parallel, bool simd, const MpiContext& mpi,
                             StepTiming* timing = nullptr)
{
    auto t0 = steady_clock_t::now();

    auto ants_t0 = steady_clock_t::now();
    ant::step_timing ant_details;
    std::vector<position_t> marked_cells;
    marked_cells.reserve(ants.size() * 2);

    if (!parallel) {
        for (std::size_t i = 0; i < ants.size(); ++i) {
            ants[i].advance(phen, land, pos_food, pos_nest, cpteur, &ant_details, &marked_cells);
        }
    } else {
#ifdef _OPENMP
        const int thread_count = omp_get_max_threads();
        std::vector<std::vector<position_t>> marked_per_thread(static_cast<std::size_t>(thread_count));
        std::vector<std::size_t> food_per_thread(static_cast<std::size_t>(thread_count), 0);
        std::vector<ant::step_timing> timing_per_thread(static_cast<std::size_t>(thread_count));

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& local_marked = marked_per_thread[static_cast<std::size_t>(tid)];
            auto& local_timing = timing_per_thread[static_cast<std::size_t>(tid)];
            std::size_t local_food = 0;
            local_marked.reserve((ants.size() / static_cast<std::size_t>(thread_count)) * 3 + 128);

#pragma omp for schedule(static)
            for (std::size_t i = 0; i < ants.size(); ++i) {
                ants[i].advance(phen, land, pos_food, pos_nest, local_food, &local_timing, &local_marked);
            }
            food_per_thread[static_cast<std::size_t>(tid)] = local_food;
        }

        for (std::size_t t = 0; t < marked_per_thread.size(); ++t) {
            cpteur += food_per_thread[t];
            ant_details.select_move_ms += timing_per_thread[t].select_move_ms;
            ant_details.terrain_cost_ms += timing_per_thread[t].terrain_cost_ms;
            ant_details.moves += timing_per_thread[t].moves;
            auto& v = marked_per_thread[t];
            marked_cells.insert(marked_cells.end(), v.begin(), v.end());
        }
#else
        for (std::size_t i = 0; i < ants.size(); ++i) {
            ants[i].advance(phen, land, pos_food, pos_nest, cpteur, &ant_details, &marked_cells);
        }
#endif
    }
    auto ants_t1 = steady_clock_t::now();

    auto mark_t0 = steady_clock_t::now();
    std::sort(marked_cells.begin(), marked_cells.end(), position_less);
    marked_cells.erase(std::unique(marked_cells.begin(), marked_cells.end(), position_equal), marked_cells.end());
    for (const auto& pos : marked_cells) {
        phen.mark_pheronome(pos);
    }
    auto mark_t1 = steady_clock_t::now();

    auto evap_t0 = steady_clock_t::now();
#ifdef USE_MPI
    if (mpi.enabled && mpi.size > 1) {
        auto sync_mark_t0 = steady_clock_t::now();
        std::vector<double> local_mark_buffer;
        std::vector<double> merged_mark_buffer;
        phen.export_buffer_flat(local_mark_buffer);
        merged_mark_buffer.resize(local_mark_buffer.size());
        MPI_Allreduce(local_mark_buffer.data(), merged_mark_buffer.data(), static_cast<int>(local_mark_buffer.size()),
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        phen.import_buffer_flat(merged_mark_buffer);
        auto sync_mark_t1 = steady_clock_t::now();
        if (timing != nullptr) {
            timing->mpi_mark_sync_ms += to_ms(sync_mark_t0, sync_mark_t1);
        }

        std::size_t begin = 0;
        std::size_t end = 0;
        split_even_work(phen.dimensions(), static_cast<std::size_t>(mpi.rank), static_cast<std::size_t>(mpi.size), begin, end);
        if (end > begin) {
            phen.do_evaporation_rows(begin + 1, end, simd);
        }

        auto sync_evap_t0 = steady_clock_t::now();
        std::vector<double> local_evap_buffer;
        std::vector<double> merged_evap_buffer;
        phen.export_evaporated_rows_flat(begin + 1, end, local_evap_buffer);
        merged_evap_buffer.resize(local_evap_buffer.size());
        MPI_Allreduce(local_evap_buffer.data(), merged_evap_buffer.data(), static_cast<int>(local_evap_buffer.size()),
                      MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        phen.import_buffer_flat(merged_evap_buffer);
        auto sync_evap_t1 = steady_clock_t::now();
        if (timing != nullptr) {
            timing->mpi_evap_sync_ms += to_ms(sync_evap_t0, sync_evap_t1);
        }
    } else {
        phen.do_evaporation(simd);
    }
#else
    (void)mpi;
    phen.do_evaporation(simd);
#endif
    auto evap_t1 = steady_clock_t::now();

    auto upd_t0 = steady_clock_t::now();
    phen.update(simd);
    auto upd_t1 = steady_clock_t::now();

    if (timing != nullptr) {
        timing->ants_ms += to_ms(ants_t0, ants_t1);
        timing->ant_select_move_ms += ant_details.select_move_ms;
        timing->ant_terrain_cost_ms += ant_details.terrain_cost_ms;
        timing->ant_mark_pheromone_ms += to_ms(mark_t0, mark_t1);
        timing->ant_moves += ant_details.moves;
        timing->evaporation_ms += to_ms(evap_t0, evap_t1);
        timing->pheromone_update_ms += to_ms(upd_t0, upd_t1);
        timing->advance_time_ms += to_ms(t0, steady_clock_t::now());
    }
}

static void advance_ant_vectorized(std::size_t i, AntSoA& ants, const pheronome& phen, const fractal_land& land,
                                   const position_t& pos_food, const position_t& pos_nest, std::size_t& cpteur_food,
                                   ant::step_timing* timing, std::vector<position_t>* marked_cells, double eps)
{
    auto& ant_seed = ants.seed[i];
    double consumed_time = 0.0;
    while (consumed_time < 1.0) {
        auto select_t0 = steady_clock_t::now();

        const int ind_pher = (ants.state[i] == AntSoA::loaded) ? 1 : 0;
        const double choice = rand_double(0., 1., ant_seed);
        const int old_x = ants.x[i];
        const int old_y = ants.y[i];
        int new_x = old_x;
        int new_y = old_y;
        const double max_phen = std::max({phen(new_x - 1, new_y)[ind_pher], phen(new_x + 1, new_y)[ind_pher],
                                          phen(new_x, new_y - 1)[ind_pher], phen(new_x, new_y + 1)[ind_pher]});
        if ((choice > eps) || (max_phen <= 0.0)) {
            do {
                new_x = old_x;
                new_y = old_y;
                const int d = rand_int32(1, 4, ant_seed);
                if (d == 1) new_x -= 1;
                if (d == 2) new_y -= 1;
                if (d == 3) new_x += 1;
                if (d == 4) new_y += 1;
            } while (phen[position_t{new_x, new_y}][ind_pher] == -1);
        } else {
            if (phen(new_x - 1, new_y)[ind_pher] == max_phen) new_x -= 1;
            else if (phen(new_x + 1, new_y)[ind_pher] == max_phen) new_x += 1;
            else if (phen(new_x, new_y - 1)[ind_pher] == max_phen) new_y -= 1;
            else new_y += 1;
        }

        auto select_t1 = steady_clock_t::now();
        if (timing != nullptr) {
            timing->select_move_ms += std::chrono::duration<double, std::milli>(select_t1 - select_t0).count();
            timing->moves += 1;
        }

        auto terrain_t0 = steady_clock_t::now();
        consumed_time += land(new_x, new_y);
        auto terrain_t1 = steady_clock_t::now();
        if (timing != nullptr) {
            timing->terrain_cost_ms += std::chrono::duration<double, std::milli>(terrain_t1 - terrain_t0).count();
        }

        if (marked_cells != nullptr) {
            marked_cells->push_back(position_t{new_x, new_y});
        }

        ants.x[i] = new_x;
        ants.y[i] = new_y;
        if (new_x == pos_nest.x && new_y == pos_nest.y) {
            if (ants.state[i] == AntSoA::loaded) cpteur_food += 1;
            ants.state[i] = AntSoA::unloaded;
        }
        if (new_x == pos_food.x && new_y == pos_food.y) {
            ants.state[i] = AntSoA::loaded;
        }
    }
}

static void advance_time_vectorized(const fractal_land& land, pheronome& phen, const position_t& pos_nest,
                                    const position_t& pos_food, AntSoA& ants, std::size_t& cpteur, bool parallel,
                                    bool simd, StepTiming* timing = nullptr, double eps = 0.8)
{
    auto t0 = steady_clock_t::now();

    auto ants_t0 = steady_clock_t::now();
    ant::step_timing ant_details;
    std::vector<position_t> marked_cells;
    marked_cells.reserve(ants.size() * 2);

    if (!parallel) {
        for (std::size_t i = 0; i < ants.size(); ++i)
            advance_ant_vectorized(i, ants, phen, land, pos_food, pos_nest, cpteur, &ant_details, &marked_cells, eps);
    } else {
#ifdef _OPENMP
        const int thread_count = omp_get_max_threads();
        std::vector<std::vector<position_t>> marked_per_thread(static_cast<std::size_t>(thread_count));
        std::vector<std::size_t> food_per_thread(static_cast<std::size_t>(thread_count), 0);
        std::vector<ant::step_timing> timing_per_thread(static_cast<std::size_t>(thread_count));

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& local_marked = marked_per_thread[static_cast<std::size_t>(tid)];
            auto& local_timing = timing_per_thread[static_cast<std::size_t>(tid)];
            std::size_t local_food = 0;
            local_marked.reserve((ants.size() / static_cast<std::size_t>(thread_count)) * 3 + 128);

#pragma omp for schedule(static)
            for (std::size_t i = 0; i < ants.size(); ++i) {
                advance_ant_vectorized(i, ants, phen, land, pos_food, pos_nest, local_food, &local_timing, &local_marked,
                                       eps);
            }
            food_per_thread[static_cast<std::size_t>(tid)] = local_food;
        }

        for (std::size_t t = 0; t < marked_per_thread.size(); ++t) {
            cpteur += food_per_thread[t];
            ant_details.select_move_ms += timing_per_thread[t].select_move_ms;
            ant_details.terrain_cost_ms += timing_per_thread[t].terrain_cost_ms;
            ant_details.moves += timing_per_thread[t].moves;
            auto& v = marked_per_thread[t];
            marked_cells.insert(marked_cells.end(), v.begin(), v.end());
        }
#else
        for (std::size_t i = 0; i < ants.size(); ++i)
            advance_ant_vectorized(i, ants, phen, land, pos_food, pos_nest, cpteur, &ant_details, &marked_cells, eps);
#endif
    }
    auto ants_t1 = steady_clock_t::now();

    auto mark_t0 = steady_clock_t::now();
    std::sort(marked_cells.begin(), marked_cells.end(), position_less);
    marked_cells.erase(std::unique(marked_cells.begin(), marked_cells.end(), position_equal), marked_cells.end());
    for (const auto& pos : marked_cells) {
        phen.mark_pheronome(pos);
    }
    auto mark_t1 = steady_clock_t::now();

    auto evap_t0 = steady_clock_t::now();
    phen.do_evaporation(simd);
    auto evap_t1 = steady_clock_t::now();

    auto upd_t0 = steady_clock_t::now();
    phen.update(simd);
    auto upd_t1 = steady_clock_t::now();

    if (timing != nullptr) {
        timing->ants_ms += to_ms(ants_t0, ants_t1);
        timing->ant_select_move_ms += ant_details.select_move_ms;
        timing->ant_terrain_cost_ms += ant_details.terrain_cost_ms;
        timing->ant_mark_pheromone_ms += to_ms(mark_t0, mark_t1);
        timing->ant_moves += ant_details.moves;
        timing->evaporation_ms += to_ms(evap_t0, evap_t1);
        timing->pheromone_update_ms += to_ms(upd_t0, upd_t1);
        timing->advance_time_ms += to_ms(t0, steady_clock_t::now());
    }
}

static void advance_time_vectorized_submap(const fractal_land& land, pheronome& phen, const position_t& pos_nest,
                                           const position_t& pos_food, AntSoA& ants, std::size_t& cpteur,
                                           std::size_t submaps, bool simd, StepTiming* timing = nullptr,
                                           double eps = 0.8)
{
    auto t0 = steady_clock_t::now();
    auto ants_t0 = steady_clock_t::now();
    const std::size_t dim = static_cast<std::size_t>(land.dimensions());

    auto reassign_t0 = steady_clock_t::now();
    std::vector<std::vector<std::size_t>> ant_bins(submaps);
    for (std::size_t p = 0; p < submaps; ++p) {
        ant_bins[p].reserve((ants.size() / std::max<std::size_t>(1, submaps)) + 64);
    }
    for (std::size_t i = 0; i < ants.size(); ++i) {
        ant_bins[owner_from_row(ants.x[i], dim, submaps)].push_back(i);
    }
    auto reassign_t1 = steady_clock_t::now();

    ant::step_timing ant_details;
    std::size_t border_crossings = 0;
    std::size_t border_marks_exchanged = 0;

    std::vector<std::vector<position_t>> owner_marks(submaps);
    for (std::size_t p = 0; p < submaps; ++p) {
        owner_marks[p].reserve((ants.size() / std::max<std::size_t>(1, submaps)) * 2 + 128);
    }

#ifdef _OPENMP
    const int thread_count = omp_get_max_threads();
    std::vector<std::vector<std::vector<position_t>>> core_per_thread(
        static_cast<std::size_t>(thread_count), std::vector<std::vector<position_t>>(submaps));
    std::vector<std::vector<std::vector<position_t>>> border_per_thread(
        static_cast<std::size_t>(thread_count), std::vector<std::vector<position_t>>(submaps));
    std::vector<std::size_t> food_per_thread(static_cast<std::size_t>(thread_count), 0);
    std::vector<ant::step_timing> timing_per_thread(static_cast<std::size_t>(thread_count));
    std::vector<std::size_t> crossing_per_thread(static_cast<std::size_t>(thread_count), 0);
    std::vector<std::size_t> exchanged_marks_per_thread(static_cast<std::size_t>(thread_count), 0);

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& local_core = core_per_thread[static_cast<std::size_t>(tid)];
        auto& local_border = border_per_thread[static_cast<std::size_t>(tid)];
        auto& local_timing = timing_per_thread[static_cast<std::size_t>(tid)];
        std::size_t local_food = 0;
        std::size_t local_crossings = 0;
        std::size_t local_exchanged_marks = 0;
        std::vector<position_t> ant_marked;
        ant_marked.reserve(16);
        for (std::size_t p = 0; p < submaps; ++p) {
            local_core[p].reserve((ants.size() / static_cast<std::size_t>(thread_count) / std::max<std::size_t>(1, submaps)) * 2 + 32);
            local_border[p].reserve((ants.size() / static_cast<std::size_t>(thread_count) / std::max<std::size_t>(1, submaps)) + 16);
        }

#pragma omp for schedule(dynamic, 1)
        for (std::size_t owner = 0; owner < submaps; ++owner) {
            const auto& bucket = ant_bins[owner];
            for (std::size_t ant_idx : bucket) {
                ant_marked.clear();
                const std::size_t before_owner = owner_from_row(ants.x[ant_idx], dim, submaps);
                advance_ant_vectorized(ant_idx, ants, phen, land, pos_food, pos_nest, local_food, &local_timing, &ant_marked,
                                       eps);
                const std::size_t after_owner = owner_from_row(ants.x[ant_idx], dim, submaps);
                if (after_owner != before_owner) {
                    local_crossings += 1;
                }
                for (const auto& pos : ant_marked) {
                    const std::size_t target_owner = owner_from_row(pos.x, dim, submaps);
                    if (target_owner == owner) {
                        local_core[owner].push_back(pos);
                    } else {
                        local_border[target_owner].push_back(pos);
                        local_exchanged_marks += 1;
                    }
                }
            }
        }
        food_per_thread[static_cast<std::size_t>(tid)] = local_food;
        crossing_per_thread[static_cast<std::size_t>(tid)] = local_crossings;
        exchanged_marks_per_thread[static_cast<std::size_t>(tid)] = local_exchanged_marks;
    }

    for (std::size_t t = 0; t < static_cast<std::size_t>(thread_count); ++t) {
        cpteur += food_per_thread[t];
        ant_details.select_move_ms += timing_per_thread[t].select_move_ms;
        ant_details.terrain_cost_ms += timing_per_thread[t].terrain_cost_ms;
        ant_details.moves += timing_per_thread[t].moves;
        border_crossings += crossing_per_thread[t];
        border_marks_exchanged += exchanged_marks_per_thread[t];
        for (std::size_t owner = 0; owner < submaps; ++owner) {
            auto& dst = owner_marks[owner];
            auto& core = core_per_thread[t][owner];
            auto& border = border_per_thread[t][owner];
            dst.insert(dst.end(), core.begin(), core.end());
            dst.insert(dst.end(), border.begin(), border.end());
        }
    }
#else
    std::vector<position_t> ant_marked;
    ant_marked.reserve(16);
    for (std::size_t owner = 0; owner < submaps; ++owner) {
        const auto& bucket = ant_bins[owner];
        for (std::size_t ant_idx : bucket) {
            ant_marked.clear();
            const std::size_t before_owner = owner_from_row(ants.x[ant_idx], dim, submaps);
            advance_ant_vectorized(ant_idx, ants, phen, land, pos_food, pos_nest, cpteur, &ant_details, &ant_marked, eps);
            const std::size_t after_owner = owner_from_row(ants.x[ant_idx], dim, submaps);
            if (after_owner != before_owner) {
                border_crossings += 1;
            }
            for (const auto& pos : ant_marked) {
                const std::size_t target_owner = owner_from_row(pos.x, dim, submaps);
                if (target_owner == owner) owner_marks[owner].push_back(pos);
                else {
                    owner_marks[target_owner].push_back(pos);
                    border_marks_exchanged += 1;
                }
            }
        }
    }
#endif
    auto ants_t1 = steady_clock_t::now();

    auto mark_t0 = steady_clock_t::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (std::size_t owner = 0; owner < submaps; ++owner) {
        auto& marks = owner_marks[owner];
        std::sort(marks.begin(), marks.end(), position_less);
        marks.erase(std::unique(marks.begin(), marks.end(), position_equal), marks.end());
        for (const auto& pos : marks) {
            phen.mark_pheronome(pos);
        }
    }
    auto mark_t1 = steady_clock_t::now();

    auto evap_t0 = steady_clock_t::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t owner = 0; owner < submaps; ++owner) {
        std::size_t begin = 0;
        std::size_t end = 0;
        split_even_work(dim, owner, submaps, begin, end);
        if (end > begin) {
            phen.do_evaporation_rows(begin + 1, end, simd);
        }
    }
    auto evap_t1 = steady_clock_t::now();

    auto upd_t0 = steady_clock_t::now();
    phen.update(simd);
    auto upd_t1 = steady_clock_t::now();

    if (timing != nullptr) {
        timing->ants_ms += to_ms(ants_t0, ants_t1);
        timing->ant_select_move_ms += ant_details.select_move_ms;
        timing->ant_terrain_cost_ms += ant_details.terrain_cost_ms;
        timing->ant_mark_pheromone_ms += to_ms(mark_t0, mark_t1);
        timing->ant_moves += ant_details.moves;
        timing->evaporation_ms += to_ms(evap_t0, evap_t1);
        timing->pheromone_update_ms += to_ms(upd_t0, upd_t1);
        timing->submap_reassign_ms += to_ms(reassign_t0, reassign_t1);
        timing->submap_border_crossings += border_crossings;
        timing->submap_border_marks_exchanged += border_marks_exchanged;
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
    std::cout << "  --vectorized <0|1>  use SoA vectorized ant storage (default: 0)\n";
    std::cout << "  --simd <0|1>        enable explicit OpenMP SIMD directives (default: 0)\n";
    std::cout << "  --submap <0|1>      sub-map ownership mode for vectorized+OpenMP path (default: 0)\n";
    std::cout << "  --parallel <0|1>    enable parallel ant update (default: 1)\n";
    std::cout << "  --mpi <0|1>         enable MPI distributed mode (default: 0)\n";
    std::cout << "  --threads <N>       OpenMP thread count, 0 means default\n";
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
        if (arg == "--parallel" && i + 1 < argc) {
            std::size_t parallel_flag = 0;
            if (!parse_size_t_arg(argv[++i], parallel_flag) || parallel_flag > 1) return false;
            opt.parallel = (parallel_flag == 1);
            continue;
        }
        if (arg == "--vectorized" && i + 1 < argc) {
            std::size_t vectorized_flag = 0;
            if (!parse_size_t_arg(argv[++i], vectorized_flag) || vectorized_flag > 1) return false;
            opt.vectorized = (vectorized_flag == 1);
            continue;
        }
        if (arg == "--simd" && i + 1 < argc) {
            std::size_t simd_flag = 0;
            if (!parse_size_t_arg(argv[++i], simd_flag) || simd_flag > 1) return false;
            opt.simd = (simd_flag == 1);
            continue;
        }
        if (arg == "--submap" && i + 1 < argc) {
            std::size_t submap_flag = 0;
            if (!parse_size_t_arg(argv[++i], submap_flag) || submap_flag > 1) return false;
            opt.submap = (submap_flag == 1);
            continue;
        }
        if (arg == "--mpi" && i + 1 < argc) {
            std::size_t mpi_flag = 0;
            if (!parse_size_t_arg(argv[++i], mpi_flag) || mpi_flag > 1) return false;
            opt.mpi = (mpi_flag == 1);
            continue;
        }
        if (arg == "--threads" && i + 1 < argc) {
            if (!parse_size_t_arg(argv[++i], opt.threads)) return false;
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

static RunTiming run_once(bool interactive_mode, std::size_t max_iterations, bool enable_render, std::size_t nb_ants,
                          bool parallel, std::size_t thread_count, bool vectorized, bool simd, bool submap,
                          const MpiContext& mpi)
{
    RunTiming stats;
    auto run_t0 = steady_clock_t::now();
    const bool is_root = (mpi.rank == 0);

    stats.mpi_enabled = mpi.enabled;
    stats.mpi_processes = static_cast<std::size_t>(mpi.size);

    if (mpi.enabled && mpi.size > 1) {
        interactive_mode = false;
        enable_render = false;
        if (parallel) {
            if (is_root) {
                std::cout << "[info] MPI premiere facon runs in distributed-memory mode only, forcing --parallel 0\n";
            }
            parallel = false;
        }
        if (submap) {
            if (is_root) {
                std::cout << "[warning] --submap is shared-memory only, forcing --submap 0 in MPI mode\n";
            }
            submap = false;
        }
        if (vectorized) {
            if (is_root) {
                std::cout << "[warning] vectorized mode is disabled in MPI mode, forcing --vectorized 0\n";
            }
            vectorized = false;
        }
    }

#ifdef _OPENMP
    if (parallel && thread_count > 0) {
        omp_set_num_threads(static_cast<int>(thread_count));
    }
    stats.parallel_enabled = parallel;
    stats.vectorized_enabled = vectorized;
    stats.simd_enabled = simd;
    stats.submap_enabled = submap;
    stats.threads = parallel ? static_cast<std::size_t>(omp_get_max_threads()) : 1;
#else
    (void)thread_count;
    stats.parallel_enabled = false;
    stats.vectorized_enabled = vectorized;
    stats.simd_enabled = false;
    stats.submap_enabled = false;
    stats.threads = 1;
    parallel = false;
#endif

    auto sdl_t0 = steady_clock_t::now();
    Uint32 sdl_flags = enable_render ? SDL_INIT_VIDEO : SDL_INIT_TIMER;
    if (SDL_Init(sdl_flags) != 0) {
        if (is_root) {
            std::cerr << "SDL_Init failed: " << SDL_GetError() << std::endl;
        }
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
    const std::size_t land_cells = static_cast<std::size_t>(land.dimensions()) * static_cast<std::size_t>(land.dimensions());
    if (simd) {
        const double* alt = land.data();
#pragma omp simd reduction(max : max_val) reduction(min : min_val)
        for (std::size_t idx = 0; idx < land_cells; ++idx) {
            max_val = std::max(max_val, alt[idx]);
            min_val = std::min(min_val, alt[idx]);
        }
    } else {
        for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
            for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j) {
                max_val = std::max(max_val, land(i, j));
                min_val = std::min(min_val, land(i, j));
            }
    }
    double delta = max_val - min_val;
    if (simd) {
        double* alt = land.data();
#pragma omp simd
        for (std::size_t idx = 0; idx < land_cells; ++idx)
            alt[idx] = (alt[idx] - min_val) / delta;
    } else {
        for (fractal_land::dim_t i = 0; i < land.dimensions(); ++i)
            for (fractal_land::dim_t j = 0; j < land.dimensions(); ++j)
                land(i, j) = (land(i, j) - min_val) / delta;
    }
    auto norm_t1 = steady_clock_t::now();
    stats.land_normalization_ms = to_ms(norm_t0, norm_t1);

    if (vectorized && enable_render) {
        if (is_root) {
            std::cout << "[warning] render is only available with non-vectorized ants, forcing --vectorized 0\n";
        }
        vectorized = false;
        stats.vectorized_enabled = false;
    }
    if (submap && !vectorized) {
        if (is_root) {
            std::cout << "[warning] --submap requires --vectorized 1, forcing --submap 0\n";
        }
        submap = false;
        stats.submap_enabled = false;
    }
    if (submap && !parallel) {
        if (is_root) {
            std::cout << "[warning] --submap requires --parallel 1, forcing --submap 0\n";
        }
        submap = false;
        stats.submap_enabled = false;
    }

    auto ants_init_t0 = steady_clock_t::now();
    ant::set_exploration_coef(eps);
    std::vector<ant> ants;
    AntSoA ants_soa;
    std::size_t ant_begin = 0;
    std::size_t ant_end = nb_ants;
    if (mpi.enabled && mpi.size > 1) {
        split_even_work(nb_ants, static_cast<std::size_t>(mpi.rank), static_cast<std::size_t>(mpi.size), ant_begin, ant_end);
    }
    const std::size_t local_ant_count = ant_end - ant_begin;
    auto gen_ant_pos = [&land](std::size_t& ant_seed) { return rand_int32(0, land.dimensions() - 1, ant_seed); };
    if (!vectorized) {
        ants.reserve(local_ant_count);
        for (std::size_t i = 0; i < local_ant_count; ++i) {
            std::size_t ant_seed = seed + (ant_begin + i) * 7919 + static_cast<std::size_t>(mpi.rank) * 131;
            ants.emplace_back(position_t{gen_ant_pos(ant_seed), gen_ant_pos(ant_seed)}, ant_seed);
        }
    } else {
        ants_soa.reserve(local_ant_count);
        for (std::size_t i = 0; i < local_ant_count; ++i) {
            std::size_t ant_seed = seed + (ant_begin + i) * 7919 + static_cast<std::size_t>(mpi.rank) * 131;
            ants_soa.push_back(position_t{gen_ant_pos(ant_seed), gen_ant_pos(ant_seed)}, ant_seed);
        }
    }
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

    std::size_t food_quantity_local = 0;
    std::size_t food_quantity_global = 0;
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

        if (!vectorized) {
            if (mpi.enabled && mpi.size > 1) {
                advance_time_mpi(land, phen, pos_nest, pos_food, ants, food_quantity_local, parallel, simd, mpi, &stats.loop_steps);
            } else {
                advance_time(land, phen, pos_nest, pos_food, ants, food_quantity_local, parallel, simd, &stats.loop_steps);
            }
        } else {
            if (submap && parallel) {
                const std::size_t submaps = std::max<std::size_t>(1, stats.threads);
                advance_time_vectorized_submap(land, phen, pos_nest, pos_food, ants_soa, food_quantity_local, submaps,
                                               simd, &stats.loop_steps, eps);
            } else {
                advance_time_vectorized(land, phen, pos_nest, pos_food, ants_soa, food_quantity_local, parallel, simd,
                                        &stats.loop_steps, eps);
            }
        }

        if (mpi.enabled && mpi.size > 1) {
#ifdef USE_MPI
            unsigned long long local_food = static_cast<unsigned long long>(food_quantity_local);
            unsigned long long global_food = 0;
            MPI_Allreduce(&local_food, &global_food, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            food_quantity_global = static_cast<std::size_t>(global_food);
#endif
        } else {
            food_quantity_global = food_quantity_local;
        }

        if (enable_render && renderer && win) {
            auto render_t0 = steady_clock_t::now();
            renderer->display(*win, food_quantity_global);
            auto render_t1 = steady_clock_t::now();
            stats.loop_steps.render_display_ms += to_ms(render_t0, render_t1);

            auto blit_t0 = steady_clock_t::now();
            win->blit();
            auto blit_t1 = steady_clock_t::now();
            stats.loop_steps.blit_ms += to_ms(blit_t0, blit_t1);
        }

        if (is_root && not_food_in_nest && food_quantity_global > 0) {
            std::cout << "La premiere nourriture est arrivee au nid a l'iteration " << it << std::endl;
            not_food_in_nest = false;
        }

        auto loop_t1 = steady_clock_t::now();
        stats.loop_steps.loop_ms += to_ms(loop_t0, loop_t1);
        stats.iterations = it;
        stats.delivered_food = food_quantity_global;

        if (!interactive_mode && it >= max_iterations)
            cont_loop = false;
    }

    SDL_Quit();
    stats.run_total_ms = to_ms(run_t0, steady_clock_t::now());

#ifdef USE_MPI
    if (mpi.enabled && mpi.size > 1) {
        auto allreduce_max = [](double local) {
            double global = local;
            MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            return global;
        };
        stats.sdl_init_ms = allreduce_max(stats.sdl_init_ms);
        stats.land_generation_ms = allreduce_max(stats.land_generation_ms);
        stats.land_normalization_ms = allreduce_max(stats.land_normalization_ms);
        stats.ant_init_ms = allreduce_max(stats.ant_init_ms);
        stats.pheromone_init_ms = allreduce_max(stats.pheromone_init_ms);
        stats.window_init_ms = allreduce_max(stats.window_init_ms);
        stats.renderer_init_ms = allreduce_max(stats.renderer_init_ms);
        stats.setup_total_ms = allreduce_max(stats.setup_total_ms);
        stats.loop_steps.event_poll_ms = allreduce_max(stats.loop_steps.event_poll_ms);
        stats.loop_steps.advance_time_ms = allreduce_max(stats.loop_steps.advance_time_ms);
        stats.loop_steps.ants_ms = allreduce_max(stats.loop_steps.ants_ms);
        stats.loop_steps.ant_select_move_ms = allreduce_max(stats.loop_steps.ant_select_move_ms);
        stats.loop_steps.ant_terrain_cost_ms = allreduce_max(stats.loop_steps.ant_terrain_cost_ms);
        stats.loop_steps.ant_mark_pheromone_ms = allreduce_max(stats.loop_steps.ant_mark_pheromone_ms);
        stats.loop_steps.evaporation_ms = allreduce_max(stats.loop_steps.evaporation_ms);
        stats.loop_steps.pheromone_update_ms = allreduce_max(stats.loop_steps.pheromone_update_ms);
        stats.loop_steps.render_display_ms = allreduce_max(stats.loop_steps.render_display_ms);
        stats.loop_steps.blit_ms = allreduce_max(stats.loop_steps.blit_ms);
        stats.loop_steps.loop_ms = allreduce_max(stats.loop_steps.loop_ms);
        stats.loop_steps.mpi_mark_sync_ms = allreduce_max(stats.loop_steps.mpi_mark_sync_ms);
        stats.loop_steps.mpi_evap_sync_ms = allreduce_max(stats.loop_steps.mpi_evap_sync_ms);
        stats.loop_steps.submap_reassign_ms = allreduce_max(stats.loop_steps.submap_reassign_ms);
        stats.run_total_ms = allreduce_max(stats.run_total_ms);

        unsigned long long local_moves = static_cast<unsigned long long>(stats.loop_steps.ant_moves);
        unsigned long long global_moves = local_moves;
        MPI_Allreduce(&local_moves, &global_moves, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        stats.loop_steps.ant_moves = static_cast<std::size_t>(global_moves);

        unsigned long long local_cross = static_cast<unsigned long long>(stats.loop_steps.submap_border_crossings);
        unsigned long long global_cross = local_cross;
        MPI_Allreduce(&local_cross, &global_cross, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
        stats.loop_steps.submap_border_crossings = static_cast<std::size_t>(global_cross);
    }
#endif
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

    csv << "run,parallel,vectorized,simd,submap,mpi,mpi_processes,threads,nb_ants,grid_dim,iterations,ant_moves,submap_border_crossings,submap_border_marks_exchanged,work_ant_calls,work_grid_cells,sdl_init_ms,setup_total_ms,"
           "land_generation_ms,land_normalization_ms,ant_init_ms,pheromone_init_ms,window_init_ms,renderer_init_ms,"
           "event_poll_ms,advance_time_ms,ants_ms,ant_select_move_ms,ant_terrain_cost_ms,ant_mark_pheromone_ms,"
           "evaporation_ms,pheromone_update_ms,mpi_mark_sync_ms,mpi_evap_sync_ms,submap_reassign_ms,render_display_ms,blit_ms,loop_ms,delivered_food,run_total_ms,"
           "ants_us_per_ant_call,ant_select_ns_per_move,ant_terrain_ns_per_move,ant_mark_ns_per_move,"
           "evaporation_ns_per_cell,update_us_per_iter\n";

    RunTiming avg;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const RunTiming& r = runs[i];
        const double ant_calls = static_cast<double>(r.nb_ants) * static_cast<double>(r.iterations);
        const double grid_cells = static_cast<double>(r.grid_dim) * static_cast<double>(r.grid_dim) * static_cast<double>(r.iterations);
        const double ant_moves = static_cast<double>(r.loop_steps.ant_moves);
        csv << (i + 1) << "," << (r.parallel_enabled ? 1 : 0) << "," << (r.vectorized_enabled ? 1 : 0) << ","
            << (r.simd_enabled ? 1 : 0) << "," << (r.submap_enabled ? 1 : 0) << ","
            << (r.mpi_enabled ? 1 : 0) << "," << r.mpi_processes << ","
            << r.threads << ","
            << r.nb_ants << "," << r.grid_dim << "," << r.iterations << "," << r.loop_steps.ant_moves << ","
            << r.loop_steps.submap_border_crossings << "," << r.loop_steps.submap_border_marks_exchanged << ","
            << ant_calls << "," << grid_cells << "," << r.sdl_init_ms << ","
            << r.setup_total_ms << ","
            << r.land_generation_ms << "," << r.land_normalization_ms << "," << r.ant_init_ms << ","
            << r.pheromone_init_ms << "," << r.window_init_ms << "," << r.renderer_init_ms << ","
            << r.loop_steps.event_poll_ms << "," << r.loop_steps.advance_time_ms << "," << r.loop_steps.ants_ms << ","
            << r.loop_steps.ant_select_move_ms << "," << r.loop_steps.ant_terrain_cost_ms << ","
            << r.loop_steps.ant_mark_pheromone_ms << "," << r.loop_steps.evaporation_ms << ","
            << r.loop_steps.pheromone_update_ms << "," << r.loop_steps.mpi_mark_sync_ms << ","
            << r.loop_steps.mpi_evap_sync_ms << "," << r.loop_steps.submap_reassign_ms << ","
            << r.loop_steps.render_display_ms << "," << r.loop_steps.blit_ms
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
        avg.loop_steps.mpi_mark_sync_ms += r.loop_steps.mpi_mark_sync_ms;
        avg.loop_steps.mpi_evap_sync_ms += r.loop_steps.mpi_evap_sync_ms;
        avg.loop_steps.submap_reassign_ms += r.loop_steps.submap_reassign_ms;
        avg.loop_steps.submap_border_crossings += r.loop_steps.submap_border_crossings;
        avg.loop_steps.submap_border_marks_exchanged += r.loop_steps.submap_border_marks_exchanged;
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
        csv << "avg" << "," << (runs.front().parallel_enabled ? 1 : 0) << ","
            << (runs.front().vectorized_enabled ? 1 : 0) << ","
            << (runs.front().simd_enabled ? 1 : 0) << "," << (runs.front().submap_enabled ? 1 : 0) << ","
            << (runs.front().mpi_enabled ? 1 : 0) << ","
            << runs.front().mpi_processes << "," << runs.front().threads << ","
            << avg_ants << "," << avg_dim << "," << avg_iter << "," << avg_moves << ","
            << (static_cast<double>(avg.loop_steps.submap_border_crossings) * inv_n) << ","
            << (static_cast<double>(avg.loop_steps.submap_border_marks_exchanged) * inv_n) << ","
            << avg_ant_calls << ","
            << avg_grid_cells << "," << avg.sdl_init_ms * inv_n << "," << avg.setup_total_ms * inv_n << ","
            << avg.land_generation_ms * inv_n << "," << avg.land_normalization_ms * inv_n << "," << avg.ant_init_ms * inv_n
            << "," << avg.pheromone_init_ms * inv_n << "," << avg.window_init_ms * inv_n << ","
            << avg.renderer_init_ms * inv_n << "," << avg.loop_steps.event_poll_ms * inv_n << ","
            << avg.loop_steps.advance_time_ms * inv_n << "," << avg.loop_steps.ants_ms * inv_n << ","
            << avg.loop_steps.ant_select_move_ms * inv_n << "," << avg.loop_steps.ant_terrain_cost_ms * inv_n << ","
            << avg.loop_steps.ant_mark_pheromone_ms * inv_n << "," << avg.loop_steps.evaporation_ms * inv_n << ","
            << avg.loop_steps.pheromone_update_ms * inv_n << "," << avg.loop_steps.mpi_mark_sync_ms * inv_n << ","
            << avg.loop_steps.mpi_evap_sync_ms * inv_n << "," << avg.loop_steps.submap_reassign_ms * inv_n << ","
            << avg.loop_steps.render_display_ms * inv_n << ","
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

    MpiContext mpi{};
#ifdef USE_MPI
    if (opt.mpi) {
        MPI_Init(&argc, &argv);
        mpi.enabled = true;
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi.rank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpi.size);
    }
#else
    if (opt.mpi) {
        std::cerr << "--mpi requires building with USE_MPI (use make mpi).\n";
        return 1;
    }
#endif

    if (!opt.benchmark) {
        run_once(true, 0, true, opt.ants, opt.parallel, opt.threads, opt.vectorized, opt.simd, opt.submap, mpi);
#ifdef USE_MPI
        if (mpi.enabled) {
            MPI_Finalize();
        }
#endif
        return 0;
    }

    std::vector<RunTiming> runs;
    runs.reserve(opt.runs);
    for (std::size_t i = 0; i < opt.runs; ++i) {
        if (mpi.rank == 0) {
            std::cout << "[benchmark] run " << (i + 1) << "/" << opt.runs << std::endl;
        }
        RunTiming r = run_once(false, opt.iterations, opt.render, opt.ants, opt.parallel, opt.threads, opt.vectorized,
                               opt.simd, opt.submap, mpi);
        if (mpi.rank == 0) {
            runs.emplace_back(r);
        }
    }

    if (mpi.rank == 0) {
        write_csv(opt.csv_path, runs);
        std::cout << "Timing CSV written to: " << opt.csv_path << std::endl;
    }

#ifdef USE_MPI
    if (mpi.enabled) {
        MPI_Finalize();
    }
#endif
    return 0;
}
