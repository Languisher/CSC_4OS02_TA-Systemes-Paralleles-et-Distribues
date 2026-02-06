# include <chrono>
# include <random>
# include <cstdlib>
# include <sstream>
# include <string>
# include <fstream>
# include <iostream>
# include <iomanip>
# include <mpi.h>
#if defined(_OPENMP)
# include <omp.h>
#endif

// Attention , ne marche qu'en C++ 11 ou supérieur :
unsigned long count_darts_in_disk(unsigned long nbSamples, int rank)
{
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();
    myclock::duration d = beginning.time_since_epoch();
    unsigned seed = static_cast<unsigned>(d.count()) ^ static_cast<unsigned>(0x9e3779b9u * (rank + 1));
    std::default_random_engine generator(seed);
    std::uniform_real_distribution <double> distribution ( -1.0 ,1.0);
    unsigned long nbDarts = 0;
    // Throw nbSamples darts in the unit square [-1 :1] x [-1 :1]
    for ( unsigned sample = 0 ; sample < nbSamples ; ++ sample ) {
        double x = distribution(generator);
        double y = distribution(generator);
        // Test if the dart is in the unit disk
        if ( x*x+y*y<=1 ) nbDarts ++;
    }
    return nbDarts;
}

double approximate_pi_sequential(unsigned long nbSamples)
{
    unsigned long nbDarts = count_darts_in_disk(nbSamples, 0);
    double ratio = double(nbDarts) / double(nbSamples);
    return 4.0 * ratio;
}

double approximate_pi_openmp(unsigned long nbSamples)
{
#if defined(_OPENMP)
    unsigned long nbDarts = 0;
#pragma omp parallel
    {
        typedef std::chrono::high_resolution_clock myclock;
        myclock::time_point beginning = myclock::now();
        myclock::duration d = beginning.time_since_epoch();
        unsigned seed = static_cast<unsigned>(d.count()) ^ static_cast<unsigned>(0x7f4a7c15u * (omp_get_thread_num() + 1));
        std::default_random_engine generator(seed);
        std::uniform_real_distribution <double> distribution ( -1.0 ,1.0);
#pragma omp for reduction(+:nbDarts)
        for (unsigned long sample = 0; sample < nbSamples; ++sample) {
            double x = distribution(generator);
            double y = distribution(generator);
            if (x*x + y*y <= 1.0) nbDarts++;
        }
    }
    double ratio = double(nbDarts) / double(nbSamples);
    return 4.0 * ratio;
#else
    return approximate_pi_sequential(nbSamples);
#endif
}

int main( int nargs, char* argv[] )
{
	// On initialise le contexte MPI qui va s'occuper :
	//    1. Créer un communicateur global, COMM_WORLD qui permet de gérer
	//       et assurer la cohésion de l'ensemble des processus créés par MPI;
	//    2. d'attribuer à chaque processus un identifiant ( entier ) unique pour
	//       le communicateur COMM_WORLD
	//    3. etc...
	MPI_Init( &nargs, &argv );
	// Pour des raisons de portabilité qui débordent largement du cadre
	// de ce cours, on préfère toujours cloner le communicateur global
	// MPI_COMM_WORLD qui gère l'ensemble des processus lancés par MPI.
	MPI_Comm globComm;
	MPI_Comm_dup(MPI_COMM_WORLD, &globComm);
	// On interroge le communicateur global pour connaître le nombre de processus
	// qui ont été lancés par l'utilisateur :
	int nbp;
	MPI_Comm_size(globComm, &nbp);
	// On interroge le communicateur global pour connaître l'identifiant qui
	// m'a été attribué ( en tant que processus ). Cet identifiant est compris
	// entre 0 et nbp-1 ( nbp étant le nombre de processus qui ont été lancés par
	// l'utilisateur )
	int rank;
	MPI_Comm_rank(globComm, &rank);
	// Création d'un fichier pour ma propre sortie en écriture :
	std::stringstream fileName;
	fileName << "Output" << std::setfill('0') << std::setw(5) << rank << ".txt";
	std::ofstream output( fileName.str().c_str() );

	unsigned long totalSamples = 40UL * 1000UL * 1000UL;
	if (nargs > 1) totalSamples = std::stoul(argv[1]);

	// Sequential reference on rank 0 only.
	if (rank == 0) {
		auto t0 = std::chrono::high_resolution_clock::now();
		double piSeq = approximate_pi_sequential(totalSamples);
		auto t1 = std::chrono::high_resolution_clock::now();
		double elapsed = std::chrono::duration<double>(t1 - t0).count();
		output << "Sequential pi=" << piSeq << " time=" << elapsed << " s\n";
		std::cout << "[SEQ] pi=" << piSeq << " time=" << elapsed << " s\n";
#if defined(_OPENMP)
		auto to0 = std::chrono::high_resolution_clock::now();
		double piOmp = approximate_pi_openmp(totalSamples);
		auto to1 = std::chrono::high_resolution_clock::now();
		double elapsedOmp = std::chrono::duration<double>(to1 - to0).count();
		output << "OpenMP pi=" << piOmp << " time=" << elapsedOmp << " s\n";
		std::cout << "[OMP] pi=" << piOmp << " time=" << elapsedOmp << " s\n";
#endif
	}

	// MPI workload split.
	unsigned long base = totalSamples / static_cast<unsigned long>(nbp);
	unsigned long rem = totalSamples % static_cast<unsigned long>(nbp);
	unsigned long localSamples = base + (rank < static_cast<int>(rem) ? 1UL : 0UL);

	MPI_Barrier(globComm);
	double startP2P = MPI_Wtime();
	unsigned long localHits = count_darts_in_disk(localSamples, rank + 17);
	unsigned long totalHitsP2P = 0;
	if (rank == 0) {
		totalHitsP2P = localHits;
		for (int src = 1; src < nbp; ++src) {
			unsigned long recvHits = 0;
			MPI_Recv(&recvHits, 1, MPI_UNSIGNED_LONG, src, 100, globComm, MPI_STATUS_IGNORE);
			totalHitsP2P += recvHits;
		}
	} else {
		MPI_Send(&localHits, 1, MPI_UNSIGNED_LONG, 0, 100, globComm);
	}
	MPI_Barrier(globComm);
	double endP2P = MPI_Wtime();
	if (rank == 0) {
		double piP2P = 4.0 * static_cast<double>(totalHitsP2P) / static_cast<double>(totalSamples);
		double elapsedP2P = endP2P - startP2P;
		output << "MPI point-to-point pi=" << piP2P << " time=" << elapsedP2P << " s\n";
		std::cout << "[MPI P2P] pi=" << piP2P << " time=" << elapsedP2P << " s\n";
	}

	// MPI collective reduction.
	MPI_Barrier(globComm);
	double startRed = MPI_Wtime();
	localHits = count_darts_in_disk(localSamples, rank + 53);
	unsigned long totalHitsRed = 0;
	MPI_Reduce(&localHits, &totalHitsRed, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, globComm);
	MPI_Barrier(globComm);
	double endRed = MPI_Wtime();
	if (rank == 0) {
		double piRed = 4.0 * static_cast<double>(totalHitsRed) / static_cast<double>(totalSamples);
		double elapsedRed = endRed - startRed;
		output << "MPI reduce pi=" << piRed << " time=" << elapsedRed << " s\n";
		std::cout << "[MPI REDUCE] pi=" << piRed << " time=" << elapsedRed << " s\n";
	}

	output.close();
	// A la fin du programme, on doit synchroniser une dernière fois tous les processus
	// afin qu'aucun processus ne se termine pendant que d'autres processus continue à
	// tourner. Si on oublie cet instruction, on aura une plantage assuré des processus
	// qui ne seront pas encore terminés.
	MPI_Finalize();
	return EXIT_SUCCESS;
}
