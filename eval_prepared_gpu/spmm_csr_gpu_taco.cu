#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <thrust/complex.h>
#include "atomic.cuh"
#include <random>
#include <string>
#include "taco.h"
#include "../test/gtest/gtest.h"
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
#ifndef TACO_TENSOR_T_DEFINED
#define TACO_TENSOR_T_DEFINED
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;
typedef struct {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  int32_t      csize;         // component size
  int32_t*     mode_ordering; // mode storage ordering
  taco_mode_t* mode_types;    // mode storage types
  uint8_t***   indices;       // tensor index data (per mode)
  uint8_t*     vals;          // tensor values
  int32_t      vals_size;     // values array size
} taco_tensor_t;
#endif
#endif

using namespace taco;

void ASSERT_TENSOR_EQ(TensorBase expected, TensorBase actual) {
  SCOPED_TRACE(std::string("expected: ") + util::toString(expected));
  SCOPED_TRACE(std::string("  actual: ") + util::toString(actual));
  ASSERT_TRUE(equals(expected, actual));
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
__device__ __host__ int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayStart] >= target) {
    return arrayStart;
  }
  int lowerBound = arrayStart; // always < target
  int upperBound = arrayEnd; // always >= target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return upperBound;
}
__device__ __host__ int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {
  if (array[arrayEnd] <= target) {
    return arrayEnd;
  }
  int lowerBound = arrayStart; // always <= target
  int upperBound = arrayEnd; // always > target
  while (upperBound - lowerBound > 1) {
    int mid = (upperBound + lowerBound) / 2;
    int midValue = array[mid];
    if (midValue < target) {
      lowerBound = mid;
    }
    else if (midValue > target) {
      upperBound = mid;
    }
    else {
      return mid;
    }
  }
  return lowerBound;
}
__global__ void taco_binarySearchBeforeBlock(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int values_per_block, int num_blocks) {
  int thread = threadIdx.x;
  int block = blockIdx.x;
  int idx = block * blockDim.x + thread;
  if (idx >= num_blocks+1) {
    return;
  }

  results[idx] = taco_binarySearchBefore(array, arrayStart, arrayEnd, idx * values_per_block);
}

__host__ int * taco_binarySearchBeforeBlockLaunch(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int values_per_block, int block_size, int num_blocks){
  int num_search_blocks = (num_blocks + 1 + block_size - 1) / block_size;
  taco_binarySearchBeforeBlock<<<num_search_blocks, block_size>>>(array, results, arrayStart, arrayEnd, values_per_block, num_blocks);
  return results;
}
__global__ void taco_binarySearchIndirectBeforeBlock(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int * __restrict__ targets, int num_blocks) {
  int thread = threadIdx.x;
  int block = blockIdx.x;
  int idx = block * blockDim.x + thread;
  if (idx >= num_blocks+1) {
    return;
  }

  results[idx] = taco_binarySearchBefore(array, arrayStart, arrayEnd, targets[idx]);
}

__host__ int * taco_binarySearchIndirectBeforeBlockLaunch(int * __restrict__ array, int * __restrict__ results, int arrayStart, int arrayEnd, int * __restrict__ targets, int block_size, int num_blocks){
  int num_search_blocks = (num_blocks + 1 + block_size - 1) / block_size;
  taco_binarySearchIndirectBeforeBlock<<<num_search_blocks, block_size>>>(array, results, arrayStart, arrayEnd, targets, num_blocks);
  return results;
}
template<typename T>
__device__ inline void atomicAddWarp(T *array, int index, T val)
{
  int leader_index = __shfl_sync(-1, index, 0);
  int mask = __ballot_sync(-1, leader_index == index);
  if(mask == -1) {
    val += __shfl_down_sync(-1, val, 16);
    val += __shfl_down_sync(-1, val, 8);
    val += __shfl_down_sync(-1, val, 4);
    val += __shfl_down_sync(-1, val, 2);
    val += __shfl_down_sync(-1, val, 1);
    if(threadIdx.x % 32 == 0) {
      atomicAdd(&array[index], val);
    }
  } else {
    atomicAdd(&array[index], val);
  }
}

__global__
void spmm_csr_gpu_tacoDeviceKernel0(taco_tensor_t * __restrict__ A, taco_tensor_t * __restrict__ B, taco_tensor_t * __restrict__ C, int32_t* i_blockStarts){
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);
  int* __restrict__ A2_crd = (int*)(A->indices[1][1]);
  float* __restrict__ A_vals = (float*)(A->vals);
  int B2_dimension = (int)(B->dimensions[1]);
  float* __restrict__ B_vals = (float*)(B->vals);
  int C1_dimension = (int)(C->dimensions[0]);
  float* __restrict__ C_vals = (float*)(C->vals);

  int32_t block = blockIdx.x;
  int32_t thread = (threadIdx.x % (32));
  int32_t warp = (threadIdx.x / 32);
  if (threadIdx.x >= 512) {
    return;
  }

  for (int32_t dense_val = 0; dense_val < 4; dense_val++) {
    int32_t k = dense_val * 32 + thread;
    float tnnz_val = 0.0;
    int32_t pA2_begin = i_blockStarts[block];
    int32_t pA2_end = i_blockStarts[(block + 1)];
    int32_t fpos1 = warp * 16;
    int32_t fposA = block * 256 + fpos1;
    int32_t i_pos = taco_binarySearchBefore(A2_pos, pA2_begin, pA2_end, fposA);
    int32_t i = i_pos;
    for (int32_t nnz = 0; nnz < 16; nnz++) {
      int32_t fpos1 = warp * 16 + nnz;
      int32_t fposA = block * 256 + fpos1;
      if (fposA >= A2_pos[A1_dimension])
        break;

      int32_t f = A2_crd[fposA];
      while (fposA == A2_pos[(i_pos + 1)]) {
        i_pos = i_pos + 1;
        i = i_pos;
      }
      int32_t iC = k * C1_dimension + i;
      int32_t kB = f * B2_dimension + k;
      tnnz_val = tnnz_val + A_vals[fposA] * B_vals[kB];
      if (fposA + 1 == A2_pos[(i_pos + 1)]) {
        atomicAdd(&C_vals[iC], tnnz_val);
        tnnz_val = 0.0;
      }
    }
    int32_t iC = k * C1_dimension + i;
    atomicAdd(&C_vals[iC], tnnz_val);
  }

}

int spmm_csr_gpu_taco(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
  int C1_dimension = (int)(C->dimensions[0]);
  int C2_dimension = (int)(C->dimensions[1]);
  float* __restrict__ C_vals = (float*)(C->vals);
  int A1_dimension = (int)(A->dimensions[0]);
  int* __restrict__ A2_pos = (int*)(A->indices[1][0]);

  int32_t* i_blockStarts = 0;
  gpuErrchk(cudaMallocManaged((void**)&i_blockStarts, sizeof(int32_t) * ((A2_pos[A1_dimension] + 255) / 256 + 1)));
  i_blockStarts = taco_binarySearchBeforeBlockLaunch(A2_pos, i_blockStarts, (int32_t) 0, A1_dimension, (int32_t) 256, (int32_t) 512, ((A2_pos[A1_dimension] + 255) / 256));

  int32_t status = cudaMemset(C_vals, 0, (C2_dimension * C1_dimension * 8));

  spmm_csr_gpu_tacoDeviceKernel0<<<(A2_pos[A1_dimension] + 255) / 256, 32 * 16>>>(A, B, C, i_blockStarts);
  cudaDeviceSynchronize();

  cudaFree(i_blockStarts);

  C->vals = (uint8_t*)C_vals;
  return 0;
}

int main() {
  std::default_random_engine gen(0);
  std::uniform_real_distribution<float> unif(0.0, 1.0);

  int NUM_I = 1021/10;
  int NUM_J = 1039/10;
  int NUM_K = 128;
  float SPARSITY = .3;
  Tensor<float> A("A", {NUM_I, NUM_J}, CSR);
  Tensor<float> B("B", {NUM_J, NUM_K}, {Dense, Dense});
  Tensor<float> C("C", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));
  Tensor<float> expected("expected", {NUM_I, NUM_K}, Format({{Dense, Dense}, {1, 0}}));

  srand(434321);
  for (int i = 0; i < NUM_I; i++) {
    for (int j = 0; j < NUM_J; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < SPARSITY) {
        A.insert({i, j}, (float) ((int) (rand_float*3/SPARSITY)));
      }
    }
  }

  for (int j = 0; j < NUM_J; j++) {
    for (int k = 0; k < NUM_K; k++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      B.insert({j, k}, (float) ((int) (rand_float*3/SPARSITY)));
    }
  }

  for (int i = 0; i < NUM_I; ++i) {
    for (int k = 0; k < NUM_K; ++k) {
      C.insert({i, k}, 0.0f);
    }
  }

  A.pack();
  B.pack();
  C.pack();

  IndexVar i("i"), j("j"), k("k");
  expected(i, k) = A(i, j) * B(j, k);
  expected.compile();
  expected.assemble();
  expected.compute();

  spmm_csr_gpu_taco(C.getTacoTensorT(), A.getTacoTensorT(), B.getTacoTensorT());

  ASSERT_TENSOR_EQ(expected, C);

  return 0;
}

