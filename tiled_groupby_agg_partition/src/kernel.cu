#include "kernel.cuh"
#include "util.cuh"
#include <cuco/detail/hash_functions/murmurhash3.cuh>
#include <cstdlib>

#define MAX_PARTITION_NUM 1024
#define BLOCK_THREADS_NUM 256

template<typename key_type, typename val_type>
__global__ void hash_into_one_third_ht(key_type *groupby_keys, 
                                       val_type *agg_vals, 
                                       key_type *ht_keys, 
                                       val_type *ht_vals, 
                                       size_t kv_num,
                                       size_t one_third_ht_sz, 
                                       size_t ht_sz, 
                                       u_int32_t *kv_indicator,
                                       val_type *ht_indicator,
                                       key_type empty_key,  
                                       int random_seed)
{
  cuco::detail::MurmurHash3_32<key_type> hf(random_seed);

  auto bx = blockIdx.x;
  auto tx = threadIdx.x;
  auto ind = bx * blockDim.x + tx;

  if (ind >= kv_num) {
    return;
  }

  auto groupby_key = groupby_keys[ind];
  // calculate insert loc
  auto insert_loc = hf(groupby_key) % ht_sz;
  if (insert_loc >= one_third_ht_sz) {
    kv_indicator[ind] = 1;
    return;
  }

  auto prev = atomicCAS(&ht_keys[insert_loc], empty_key, groupby_key);

  if (prev == empty_key || prev == groupby_key)
  {
    auto sum_val = agg_vals[ind];
    if (prev == empty_key) {
      ht_indicator[insert_loc] = 1;
    }
    atomicAdd(&ht_vals[insert_loc], sum_val);
  }
  else {
    kv_indicator[ind] = 1;
  }

}


template<typename key_type, typename val_type>
__global__ void collect_one_third_ht_result(key_type *ht_keys,
                                            val_type *ht_vals,
                                            key_type *collect_keys,
                                            val_type *collect_vals, 
                                            size_t one_third_ht_sz,  
                                            key_type *ht_indicator_scan,
                                            key_type empty_key)
{
  auto bx = blockIdx.x;
  auto tx = threadIdx.x;
  auto ind = bx * blockDim.x + tx;

  if (ind >= one_third_ht_sz) {
    return;
  }

  auto ht_key = ht_keys[ind];
  if (ht_key == empty_key) {
    return;
  }

  auto ht_val = ht_vals[ind];
  auto collect_loc = ht_indicator_scan[ind] - 1; // milus 1 because of inclusive scan
  collect_keys[collect_loc] = ht_key;
  collect_vals[collect_loc] = ht_val;
}

template<typename key_type, typename val_type>
__global__ void collect_unhashed_kv_into_ht(key_type *groupby_keys,
                                            val_type *agg_vals, 
                                            key_type *ht_keys, 
                                            val_type *ht_vals, 
                                            u_int32_t *kv_indicator_scan, 
                                            size_t kv_num)
{
  auto bx = blockIdx.x;
  auto tx = threadIdx.x;
  auto ind = bx * blockDim.x + tx;

  // indicator = 1 mean unhashed kv
  u_int32_t indicator;
  if (ind == 0) {
    indicator = kv_indicator_scan[0];
  }
  else {
    indicator = kv_indicator_scan[ind] - kv_indicator_scan[ind - 1];
  }
  if (ind >= kv_num || indicator == 0) {
    return ;
  }

  auto groupby_key = groupby_keys[ind];
  auto sum_val = agg_vals[ind];
  auto collect_loc = kv_indicator_scan[ind] - 1;
  ht_keys[collect_loc] = groupby_key;
  ht_vals[collect_loc] = sum_val;  
}

template<typename key_type, typename val_type>
__global__ void kv_hash_agg_into_ht(key_type *groupby_keys,
                                    val_type *agg_vals, 
                                    key_type *ht_keys, 
                                    val_type *ht_vals, 
                                    size_t kv_num, 
                                    size_t ht_sz, 
                                    u_int32_t *ht_indicator,
                                    key_type empty_key, 
                                    int random_seed)
{
  cuco::detail::MurmurHash3_32<key_type> hf(random_seed);

  auto bx = blockIdx.x;
  auto tx = threadIdx.x;
  auto ind = bx * blockDim.x + tx;

  if (ind >= kv_num) {
    return ;
  }
    
  auto groupby_key = groupby_keys[ind];
  auto sum_val = agg_vals[ind];

  // calculate insert loc
  auto insert_loc = hf(groupby_key) % ht_sz;

  while (1)
  {
    auto prev = atomicCAS(&ht_keys[insert_loc], empty_key, groupby_key);

    if (prev == empty_key || prev == groupby_key)
    {
      if (prev == empty_key) {
        ht_indicator[insert_loc] = 1;
      }
      atomicAdd(&ht_vals[insert_loc], sum_val);
      return ;
    }
    // probe next loc
    insert_loc = (insert_loc + 1) % ht_sz;
  }
}

template<typename key_type, typename val_type>
__global__ void combine_ht_result_with_previous_result(key_type *ht_keys, 
                                                       val_type *ht_vals,
                                                       key_type *collect_keys, 
                                                       val_type *collect_vals, 
                                                       size_t ht_sz, 
                                                       u_int32_t *ht_indicator_scan,
                                                       key_type empty_key) 
{
  auto bx = blockIdx.x;
  auto tx = threadIdx.x;
  auto ind = bx * blockDim.x + tx;

  if (ind >= ht_sz) return;

  auto ht_key = ht_keys[ind];
  if (ht_key == empty_key) {
    return ;
  }

  auto ht_val = ht_vals[ind];

  auto collect_loc = ht_indicator_scan[ind] - 1;

  collect_keys[collect_loc] = ht_key;
  collect_vals[collect_loc] = ht_val;
}


template<typename key_type>
__global__ void cal_kv_intra_group_ind(key_type *keys,
                                       u_int32_t *indicator, 
                                       size_t kv_num,
                                       u_int32_t *par_rec_num, 
                                       size_t P) 
{
  cuco::detail::MurmurHash3_32<key_type> hf;

  __shared__ u_int32_t par_hist[MAX_PARTITION_NUM];
  __shared__ u_int32_t local_indicator[BLOCK_THREADS_NUM];

  auto bx = blockIdx.x;
  auto tx = threadIdx.x;
  auto ind = bx * blockDim.x + tx;
  auto bdx = blockDim.x;

  // if (ind >= kv_num) {
  //   return;
  // }

  auto ipt = (P + bdx - 1) / bdx;

  for (int i = 0; i < ipt; i++) {
    auto par_ind = tx + i * bdx;
    if (par_ind >= P) break;
    par_hist[par_ind] = 0;
  }

  __syncthreads();

  key_type key;
  size_t par_ind;
  if (ind < kv_num)
  {
    key = keys[ind];
    par_ind = hf(key) % P;
    local_indicator[tx] = atomicAdd(&par_hist[par_ind], 1);
  }

  // auto key = keys[ind];
  // auto par_ind = hf(key) % P;
  // local_indicator[tx] = atomicAdd(&par_hist[par_ind], 1);

  __syncthreads();

  for (int i = 0; i < ipt; i++) {
    auto par_ind = tx + i * bdx;
    if (par_ind >= P) break;
    par_hist[par_ind] = atomicAdd(&par_rec_num[par_ind], par_hist[par_ind]);
  }

  __syncthreads();

  // local_indicator[tx] += par_hist[par_ind];
  // indicator[ind] = local_indicator[tx] + par_hist[par_ind];

  if (ind < kv_num) {
    indicator[ind] = local_indicator[tx] + par_hist[par_ind];
  }
}

template<typename key_type, typename val_type>
__global__ void kv_inter_group_insert(key_type *keys,
                                      val_type *vals,
                                      key_type *partition_keys,
                                      val_type *partition_vals,
                                      u_int32_t *indicator, 
                                      size_t kv_num,
                                      u_int32_t *par_rec_num_scan, 
                                      size_t P)
{
  cuco::detail::MurmurHash3_32<key_type> hf;
  __shared__ u_int32_t par_exlusive_scan[MAX_PARTITION_NUM];
  auto bx = blockIdx.x;
  auto tx = threadIdx.x;
  auto ind = bx * blockDim.x + tx;
  auto bdx = blockDim.x;

  // if (ind >= kv_num) {
  //   return;
  // }

  auto ipt = (P + bdx - 1) / bdx;

  for (int i = 0; i < ipt; i++) {
    auto par_ind = tx + i * bdx;
    if (par_ind >= P) break;
    par_exlusive_scan[par_ind] = par_rec_num_scan[par_ind];
  }

  __syncthreads();

  if (ind < kv_num)
  {
    auto key = keys[ind];
    auto par_ind = hf(key) % P;
    auto inter_group_ind = indicator[ind] + par_exlusive_scan[par_ind];   
    auto val = vals[ind];
    partition_keys[inter_group_ind] = key;
    partition_vals[inter_group_ind] = val;
  }
}


__global__ void groupby_agg_and_statistic_kernel(key_type *groupby_keys, 
                                                 val_type *agg_vals, 
                                                 key_type *ht_keys, 
                                                 val_type *ht_vals,
                                                 u_int32_t *indicator,
                                                 u_int32_t *block_result_num,
                                                 size_t kv_num, 
                                                 size_t Capacity,
                                                 key_type empty_key) 
{

  // for record increaded group from this thread block
  __shared__ uint32_t increased_group_counter;
  //

  cuco::detail::MurmurHash3_32<key_type> hf;

  auto bx = blockIdx.x;
  auto tx = threadIdx.x;
  auto kv_ind = bx * blockDim.x + tx;

  if (tx == 0) {
    increased_group_counter = 0;
  }
  __syncthreads();

  if (kv_ind >= kv_num)
  {
    return;
  }

  auto groupby_key = groupby_keys[kv_ind];
  auto agg_val = agg_vals[kv_ind];
  auto insert_loc = hf(groupby_key) % Capacity;

  while (1) {
    auto prev = atomicCAS(&ht_keys[insert_loc], empty_key, groupby_key);

    if (prev == empty_key) {
      indicator[insert_loc] = 1;
      atomicAdd(&ht_vals[insert_loc], agg_val);
      atomicAdd(&increased_group_counter, 1);
      break;
    }

    if (prev == groupby_key) {
      atomicAdd(&ht_vals[insert_loc], agg_val);
      break;
    }

    insert_loc = (insert_loc + 1) % Capacity;
  }

  __syncthreads();
  if (tx == 0) {
    block_result_num[bx] = increased_group_counter;
  }
}


__global__ void device_reduce_kernel(u_int32_t *reduce_array, u_int32_t n, u_int32_t *sum) {
  __shared__ uint32_t ras[BLOCK_THREADS_NUM];
  auto tx = threadIdx.x;
  auto bx = blockIdx.x;
  auto segment = 2 * blockDim.x *bx;
  auto ind = segment + tx;
  ras[tx] = 0;
  if (ind < n) {
    ras[tx] = reduce_array[ind];
  }
  if (ind + BLOCK_THREADS_NUM < n) {
    ras[tx] += reduce_array[ind + BLOCK_THREADS_NUM];
  }

  for (u_int32_t stride = BLOCK_THREADS_NUM / 2; stride >= 1; stride /= 2) {
    __syncthreads();
    if (tx < stride) {
      ras[tx] += ras[tx + stride];
    }
  }

  if (tx == 0) {
    atomicAdd(sum, ras[0]);
  }
}

__global__ void collect_kv_in_ht(key_type *ht_keys,
                                 val_type *ht_vals,
                                 u_int32_t *indicator,
                                 key_type *collect_keys,
                                 val_type *collect_vals,
                                 size_t Capacity,
                                 key_type empty_key)
{
  auto bx = blockIdx.x;
  auto tx = threadIdx.x;
  auto ind = bx * blockDim.x + tx;

  if (ind >= Capacity)
  {
    return;
  }
  auto ht_key = ht_keys[ind];
  if (ht_key == empty_key)
  {
    return;
  }

  auto ht_val = ht_vals[ind];
  auto collect_loc = indicator[ind];
  collect_keys[collect_loc] = ht_key;
  collect_vals[collect_loc] = ht_val;
}

template 
__global__ void hash_into_one_third_ht(u_int32_t *, 
                                       u_int32_t *, 
                                       u_int32_t *, 
                                       u_int32_t *, 
                                       size_t,
                                       size_t, 
                                       size_t, 
                                       u_int32_t *,
                                       u_int32_t *,
                                       u_int32_t,  
                                       int);

template 
__global__ void collect_one_third_ht_result(u_int32_t *ht_keys,
                                            u_int32_t *ht_vals,
                                            u_int32_t *collect_keys,
                                            u_int32_t *collect_vals, 
                                            size_t one_third_ht_sz,  
                                            u_int32_t *ht_indicator_scan,
                                            u_int32_t empty_key);

template 
__global__ void collect_unhashed_kv_into_ht(u_int32_t *groupby_keys,
                                            u_int32_t *agg_vals, 
                                            u_int32_t *ht_keys, 
                                            u_int32_t *ht_vals, 
                                            u_int32_t *kv_indicator_scan, 
                                            size_t kv_num);

template
__global__ void kv_hash_agg_into_ht(u_int32_t *groupby_keys,
                                    u_int32_t *agg_vals, 
                                    u_int32_t *ht_keys, 
                                    u_int32_t *ht_vals, 
                                    size_t kv_num, 
                                    size_t ht_sz, 
                                    u_int32_t *ht_indicator,
                                    u_int32_t empty_key, 
                                    int random_seed);

template
__global__ void combine_ht_result_with_previous_result(u_int32_t *ht_keys, 
                                                       u_int32_t *ht_vals,
                                                       u_int32_t *collect_keys, 
                                                       u_int32_t *collect_vals, 
                                                       size_t ht_sz, 
                                                       u_int32_t *ht_indicator_scan,
                                                       u_int32_t empty_key);

template
__global__ void cal_kv_intra_group_ind(key_type *keys,
                                       u_int32_t *indicator, 
                                       size_t kv_num,
                                       u_int32_t *par_rec_num, 
                                       size_t P);

template
__global__ void kv_inter_group_insert(key_type *keys,
                                      val_type *vals,
                                      key_type *partition_keys,
                                      val_type *partition_vals,
                                      u_int32_t *indicator, 
                                      size_t kv_num,
                                      u_int32_t *par_rec_num_scan, 
                                      size_t P);