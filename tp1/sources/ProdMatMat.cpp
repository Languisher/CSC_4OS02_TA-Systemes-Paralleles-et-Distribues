#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"

namespace {
prod_algo g_algo = naive;
int g_block_size = 64;

void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock,
                   const Matrix& A, const Matrix& B, Matrix& C) {
  const int iEnd = std::min(A.nbRows, iRowBlkA + szBlock);
  const int jEnd = std::min(B.nbCols, iColBlkB + szBlock);
  const int kEnd = std::min(A.nbCols, iColBlkA + szBlock);
  for (int i = iRowBlkA; i < iEnd; ++i) {
    for (int k = iColBlkA; k < kEnd; ++k) {
      const double aik = A(i, k);
      for (int j = iColBlkB; j < jEnd; ++j) {
        C(i, j) += aik * B(k, j);
      }
    }
  }
}

void prodNaive(const Matrix& A, const Matrix& B, Matrix& C) {
  for (int i = 0; i < A.nbRows; ++i) {
    for (int k = 0; k < A.nbCols; ++k) {
      const double aik = A(i, k);
      for (int j = 0; j < B.nbCols; ++j) {
        C(i, j) += aik * B(k, j);
      }
    }
  }
}

void prodIJK(const Matrix& A, const Matrix& B, Matrix& C) {
  for (int i = 0; i < A.nbRows; ++i) {
    for (int j = 0; j < B.nbCols; ++j) {
      for (int k = 0; k < A.nbCols; ++k) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

void prodJIK(const Matrix& A, const Matrix& B, Matrix& C) {
  for (int j = 0; j < B.nbCols; ++j) {
    for (int i = 0; i < A.nbRows; ++i) {
      for (int k = 0; k < A.nbCols; ++k) {
        C(i, j) += A(i, k) * B(k, j);
      }
    }
  }
}

void prodIKJ(const Matrix& A, const Matrix& B, Matrix& C) {
  for (int i = 0; i < A.nbRows; ++i) {
    for (int k = 0; k < A.nbCols; ++k) {
      const double aik = A(i, k);
      for (int j = 0; j < B.nbCols; ++j) {
        C(i, j) += aik * B(k, j);
      }
    }
  }
}

void prodKIJ(const Matrix& A, const Matrix& B, Matrix& C) {
  for (int k = 0; k < A.nbCols; ++k) {
    for (int i = 0; i < A.nbRows; ++i) {
      const double aik = A(i, k);
      for (int j = 0; j < B.nbCols; ++j) {
        C(i, j) += aik * B(k, j);
      }
    }
  }
}

void prodJKI(const Matrix& A, const Matrix& B, Matrix& C) {
  for (int j = 0; j < B.nbCols; ++j) {
    for (int k = 0; k < A.nbCols; ++k) {
      const double bkj = B(k, j);
      for (int i = 0; i < A.nbRows; ++i) {
        C(i, j) += A(i, k) * bkj;
      }
    }
  }
}

void prodKJI(const Matrix& A, const Matrix& B, Matrix& C) {
  for (int k = 0; k < A.nbCols; ++k) {
    for (int j = 0; j < B.nbCols; ++j) {
      const double bkj = B(k, j);
      for (int i = 0; i < A.nbRows; ++i) {
        C(i, j) += A(i, k) * bkj;
      }
    }
  }
}

void prodParallelNaive(const Matrix& A, const Matrix& B, Matrix& C) {
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
  for (int j = 0; j < B.nbCols; ++j) {
    for (int k = 0; k < A.nbCols; ++k) {
      const double bkj = B(k, j);
      for (int i = 0; i < A.nbRows; ++i) {
        C(i, j) += A(i, k) * bkj;
      }
    }
  }
#else
  prodNaive(A, B, C);
#endif
}

void prodBlock(const Matrix& A, const Matrix& B, Matrix& C, int blockSize) {
  const int bs = std::max(1, blockSize);
  for (int iBlk = 0; iBlk < A.nbRows; iBlk += bs) {
    for (int kBlk = 0; kBlk < A.nbCols; kBlk += bs) {
      for (int jBlk = 0; jBlk < B.nbCols; jBlk += bs) {
        prodSubBlocks(iBlk, jBlk, kBlk, bs, A, B, C);
      }
    }
  }
}

void prodParallelBlockOverJ(const Matrix& A, const Matrix& B, Matrix& C, int blockSize) {
#if defined(_OPENMP)
  const int bs = std::max(1, blockSize);
#pragma omp parallel for schedule(static)
  for (int jBlk = 0; jBlk < B.nbCols; jBlk += bs) {
    for (int kBlk = 0; kBlk < A.nbCols; kBlk += bs) {
      for (int iBlk = 0; iBlk < A.nbRows; iBlk += bs) {
        prodSubBlocks(iBlk, jBlk, kBlk, bs, A, B, C);
      }
    }
  }
#else
  prodBlock(A, B, C, blockSize);
#endif
}

void prodParallelBlockCollapsed(const Matrix& A, const Matrix& B, Matrix& C, int blockSize) {
#if defined(_OPENMP)
  const int bs = std::max(1, blockSize);
#pragma omp parallel for collapse(2) schedule(static)
  for (int iBlk = 0; iBlk < A.nbRows; iBlk += bs) {
    for (int jBlk = 0; jBlk < B.nbCols; jBlk += bs) {
      for (int kBlk = 0; kBlk < A.nbCols; kBlk += bs) {
        prodSubBlocks(iBlk, jBlk, kBlk, bs, A, B, C);
      }
    }
  }
#else
  prodBlock(A, B, C, blockSize);
#endif
}
}  // namespace

Matrix operator*(const Matrix& A, const Matrix& B) {
  assert(A.nbCols == B.nbRows);
  Matrix C(A.nbRows, B.nbCols, 0.0);
  switch (g_algo) {
    case naive:
      prodNaive(A, B, C);
      break;
    case ijk:
      prodIJK(A, B, C);
      break;
    case jik:
      prodJIK(A, B, C);
      break;
    case ikj:
      prodIKJ(A, B, C);
      break;
    case kij:
      prodKIJ(A, B, C);
      break;
    case jki:
      prodJKI(A, B, C);
      break;
    case kji:
      prodKJI(A, B, C);
      break;
    case block:
      prodBlock(A, B, C, g_block_size);
      break;
    case parallel_naive:
      prodParallelNaive(A, B, C);
      break;
    case parallel_block1:
      prodParallelBlockOverJ(A, B, C, g_block_size);
      break;
    case parallel_block2:
      prodParallelBlockCollapsed(A, B, C, g_block_size);
      break;
  }
  return C;
}

void setProdMatMat(prod_algo algo) { g_algo = algo; }

void setBlockSize(int size) { g_block_size = std::max(1, size); }

void setNbThreads(int n) {
#if defined(_OPENMP)
  if (n > 0) omp_set_num_threads(n);
#else
  (void)n;
#endif
}
