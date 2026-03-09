#include "ant.hpp"
#include <chrono>
#include <iostream>
#include "rand_generator.hpp"

double ant::m_eps = 0.;

using steady_clock_t = std::chrono::steady_clock;

void ant::advance( pheronome& phen, const fractal_land& land, const position_t& pos_food, const position_t& pos_nest,
                   std::size_t& cpteur_food, step_timing* timing ) 
{
    auto ant_choice = [this]() mutable { return rand_double( 0., 1., this->m_seed ); };
    auto dir_choice = [this]() mutable { return rand_int32( 1, 4, this->m_seed ); };
    double                                   consumed_time = 0.;
    // Tant que la fourmi peut encore bouger dans le pas de temps imparti
    while ( consumed_time < 1. ) {
        auto select_t0 = steady_clock_t::now();
        // Si la fourmi est chargée, elle suit les phéromones de deuxième type, sinon ceux du premier.
        int        ind_pher    = ( is_loaded( ) ? 1 : 0 );
        double     choix       = ant_choice( );
        position_t old_pos_ant = get_position( );
        position_t new_pos_ant = old_pos_ant;
        double max_phen    = std::max( {phen( new_pos_ant.x - 1, new_pos_ant.y )[ind_pher],
                                     phen( new_pos_ant.x + 1, new_pos_ant.y )[ind_pher],
                                     phen( new_pos_ant.x, new_pos_ant.y - 1 )[ind_pher],
                                     phen( new_pos_ant.x, new_pos_ant.y + 1 )[ind_pher]} );
        if ( ( choix > m_eps ) || ( max_phen <= 0. ) ) {
            do {
                new_pos_ant = old_pos_ant;
                int d = dir_choice();
                if ( d==1 ) new_pos_ant.x  -= 1;
                if ( d==2 ) new_pos_ant.y -= 1;
                if ( d==3 ) new_pos_ant.x  += 1;
                if ( d==4 ) new_pos_ant.y += 1;

            } while ( phen[new_pos_ant][ind_pher] == -1 );
        } else {
            // On choisit la case où le phéromone est le plus fort.
            if ( phen( new_pos_ant.x - 1, new_pos_ant.y )[ind_pher] == max_phen )
                new_pos_ant.x -= 1;
            else if ( phen( new_pos_ant.x + 1, new_pos_ant.y )[ind_pher] == max_phen )
                new_pos_ant.x += 1;
            else if ( phen( new_pos_ant.x, new_pos_ant.y - 1 )[ind_pher] == max_phen )
                new_pos_ant.y -= 1;
            else  // if (phen(new_pos_ant.first,new_pos_ant.second+1)[ind_pher] == max_phen)
                new_pos_ant.y += 1;
        }
        auto select_t1 = steady_clock_t::now();
        if ( timing != nullptr ) {
            timing->select_move_ms += std::chrono::duration<double, std::milli>(select_t1 - select_t0).count();
            timing->moves += 1;
        }

        auto terrain_t0 = steady_clock_t::now();
        consumed_time += land( new_pos_ant.x, new_pos_ant.y );
        auto terrain_t1 = steady_clock_t::now();
        if ( timing != nullptr ) {
            timing->terrain_cost_ms += std::chrono::duration<double, std::milli>(terrain_t1 - terrain_t0).count();
        }

        auto mark_t0 = steady_clock_t::now();
        phen.mark_pheronome( new_pos_ant );
        auto mark_t1 = steady_clock_t::now();
        if ( timing != nullptr ) {
            timing->mark_pheromone_ms += std::chrono::duration<double, std::milli>(mark_t1 - mark_t0).count();
        }

        m_position = new_pos_ant;
        if ( get_position( ) == pos_nest ) {
            if ( is_loaded( ) ) {
                cpteur_food += 1;
            }
            unset_loaded( );
        }
        if ( get_position( ) == pos_food ) {
            set_loaded( );
        }
    }
}
