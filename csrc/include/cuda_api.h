#pragma once

#include <torch/python.h>

#define DEBUG 1

#ifdef DEBUG

// NOTE:tensor malloc as device before we call
// e.g. data.to("cuda") in python
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CUDA_ERROR_CHECK(condition)                                            \
  do {                                                                         \
    cudaError_t error = condition;                                             \
    if (error != cudaSuccess) {                                                \
      printf("CUDA_CHECK error in line %d of file %s \
              : %s \n",                                                        \
             __LINE__, __FILE__, cudaGetErrorString(error));      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#else

#define CHECK_CUDA(x) do { } while (0)
#define CHECK_CONTIGUOUS(x) do { } while (0)
#define CHECK_INPUT(x) do { } while (0)
#define CUDA_ERROR_CHECK(condition) do { condition; } while (0)

#endif // DEBUG


