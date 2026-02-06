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

  int token = 0;
  const int next = (rank + 1) % nbp;
  const int prev = (rank - 1 + nbp) % nbp;

  if (rank == 0) {
    token = 1;
    MPI_Send(&token, 1, MPI_INT, next, 42, comm);
    MPI_Recv(&token, 1, MPI_INT, prev, 42, comm, MPI_STATUS_IGNORE);
    std::cout << "Final token value at rank 0: " << token << "\n";
  } else {
    MPI_Recv(&token, 1, MPI_INT, prev, 42, comm, MPI_STATUS_IGNORE);
    token += 1;
    MPI_Send(&token, 1, MPI_INT, next, 42, comm);
  }

  MPI_Comm_free(&comm);
  MPI_Finalize();
  return EXIT_SUCCESS;
}
