#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <vector>
#include <string>
#include <cmath>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>

#include "data_generator.hpp"
#include "group_then_partition.cuh"
#include "groupby_agg_intra_partition.cuh"
#include "CLI11.hpp"

typedef u_int32_t k_type;
typedef u_int32_t v_type;

void query_gpu_info() 
{
  int devCount;
  int gpu_ind = 0;
  cudaGetDeviceCount(&devCount);
  // std::cout << "devCount:" << devCount << "\n";

  cudaDeviceProp devProp;
  for (int i = 0; i < devCount; i++)
  {
    cudaGetDeviceProperties(&devProp, i);
    // std::cout << "name:" << devProp.name << "\n";
    // std::cout << "major:" << devProp.major << "\n";
  }

  cudaSetDevice(gpu_ind);
  // std::cout << "cudaSetDevice gpu_ind: " << gpu_ind << "\n\n";
}

__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        C[tid] = A[tid] + B[tid];
    }
}

void gpu_warm_up() {
  const int N = 1024;  // Number of elements in arrays
  const int threadsPerBlock = 256;
  const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;

  // Allocate memory on host
  float *h_A = new float[N];
  float *h_B = new float[N];
  float *h_C = new float[N];

  // Initialize input arrays
  for (int i = 0; i < N; ++i) {
    h_A[i] = static_cast<float>(i);
    h_B[i] = std::sqrt(static_cast<float>(i));
  }

  // Allocate memory on device (GPU)
  float *d_A, *d_B, *d_C;
  cudaMalloc(&d_A, N * sizeof(float));
  cudaMalloc(&d_B, N * sizeof(float));
  cudaMalloc(&d_C, N * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the kernel
  vectorAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);

  // Copy result back to host
  cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Clean up
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
}

int main(int argc, char *argv[])
{
  query_gpu_info();

  CLI::App app;
  k_type kt_max_version_1 = 0xffffffff;
  k_type kt_max = 0xffffffff;
  v_type vt_max = 0xffffffff;
  size_t kv_num = 1e9;
  size_t cardinality_percentage = 10;
  int dist_kind = 0;
  size_t tile_len = 1e7;
  size_t P = 300;
  size_t nstreams = 4;
  app.add_option("-n", kv_num);
  app.add_option("-c", cardinality_percentage);
  app.add_option("-d", dist_kind);
  app.add_option("-l", tile_len);
  CLI11_PARSE(app, argc, argv);

  auto Capacity = tile_len;
  size_t min_load_num = Capacity * 3 / 5;
  size_t max_load_num = Capacity * 4 / 5;

  
  size_t cardinality = kv_num * cardinality_percentage / 100;
  double skew_factor = 0.9;

  k_type *host_keys;
  v_type *host_vals;

  cudaMallocHost(&host_keys, sizeof(k_type) * kv_num);
  cudaMallocHost(&host_vals, sizeof(v_type) * kv_num);

  /// generate kv
  // std::srand(std::time(nullptr));
  // std::random_device r;
  // std::default_random_engine generator(r());
  // std::default_random_engine generator;
  // k_type empty_key = 0xffffffff;
  // generate_various_dist_kv_array<k_type, v_type>(host_keys, host_vals, cardinality, kv_num, skew_factor, generator, dist_kind, empty_key);
  auto start_time = std::chrono::steady_clock::now();
  generate_various_dist_kv_set_multithread_version_2<k_type, v_type>(host_keys,
                                                                      host_vals,
                                                                      cardinality,
                                                                      kv_num,
                                                                      skew_factor,
                                                                      dist_kind,
                                                                      kt_max,
                                                                      vt_max);
  auto end_time = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  std::cout << "generate_various_dist_kv_set_multithread kv_num " <<kv_num << " cardinality " << cardinality << " elapsed time: " << duration.count() << " microseconds\n";
  
  ///

  std::vector<par_result> par_result_vec(P);
  gpu_warm_up();
  
  groupby_agg_partition<k_type, v_type>(host_keys,
                                        host_vals,
                                        kv_num,
                                        tile_len,
                                        P,
                                        par_result_vec,
                                        nstreams);
  
  
  key_type *host_groupby_keys_result;
  val_type *host_agg_vals_result;
  std::vector<size_t> par_kv_begin;
  std::vector<size_t> par_result_kv_num;

  groupby_agg_intra_partition(par_result_vec,
                              host_groupby_keys_result,
                              host_agg_vals_result,
                              Capacity,
                              min_load_num,
                              max_load_num,
                              nstreams,
                              par_kv_begin,
                              par_result_kv_num);

  size_t result_kv_num = 0;
  for (size_t i=0; i<par_result_kv_num.size(); i++)
  {
    result_kv_num += par_result_kv_num[i];
  }
  printf("result kv num: %ld\n", result_kv_num);
  
}