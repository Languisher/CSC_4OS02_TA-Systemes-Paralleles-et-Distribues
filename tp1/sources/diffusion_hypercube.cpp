#include <cmath>
#include <cstdlib>
#include <iostream>
#include <mpi.h>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &comm);

  int rank = 0;
  int nbp = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nbp);

  int dim = 0;
  if (argc > 1) dim = std::atoi(argv[1]);
  if (dim <= 0) {
    int p = nbp;
    while ((1 << dim) < p) ++dim;
  }

  const int expected = 1 << dim;
  if (nbp != expected) {
    if (rank == 0) {
      std::cerr << "Error: this program requires exactly 2^d processes.\n";
      std::cerr << "Given d=" << dim << ", expected " << expected << ", got " << nbp << ".\n";
    }
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return EXIT_FAILURE;
  }

  int token = 0;
  if (rank == 0) token = 1234;

  for (int k = 0; k < dim; ++k) {
    int partner = rank ^ (1 << k);
    if ((rank & (1 << k)) == 0) {
      MPI_Send(&token, 1, MPI_INT, partner, 99 + k, comm);
    } else {
      MPI_Recv(&token, 1, MPI_INT, partner, 99 + k, comm, MPI_STATUS_IGNORE);
    }
  }

  std::cout << "rank " << rank << " received token " << token << "\n";

  MPI_Comm_free(&comm);
  MPI_Finalize();
  return EXIT_SUCCESS;
}
