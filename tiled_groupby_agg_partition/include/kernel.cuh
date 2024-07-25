#pragma once
#include <cstdlib>

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
                                       int random_seed);

template<typename key_type, typename val_type>
__global__ void collect_one_third_ht_result(key_type *ht_keys,
                                            val_type *ht_vals,
                                            key_type *collect_keys,
                                            val_type *collect_vals, 
                                            size_t one_third_ht_sz,  
                                            key_type *ht_indicator_scan,
                                            key_type empty_key);

template<typename key_type, typename val_type>
__global__ void collect_unhashed_kv_into_ht(key_type *groupby_keys,
                                            val_type *agg_vals, 
                                            key_type *ht_keys, 
                                            val_type *ht_vals, 
                                            u_int32_t *kv_indicator_scan, 
                                            size_t kv_num);

template<typename key_type, typename val_type>
__global__ void kv_hash_agg_into_ht(key_type *groupby_keys,
                                    val_type *agg_vals, 
                                    key_type *ht_keys, 
                                    val_type *ht_vals, 
                                    size_t kv_num, 
                                    size_t ht_sz, 
                                    u_int32_t *ht_indicator,
                                    key_type empty_key, 
                                    int random_seed);

template<typename key_type, typename val_type>
__global__ void combine_ht_result_with_previous_result(key_type *ht_keys, 
                                                       val_type *ht_vals,
                                                       key_type *collect_keys, 
                                                       val_type *collect_vals, 
                                                       size_t ht_sz, 
                                                       u_int32_t *ht_indicator_scan,
                                                       key_type empty_key);

template<typename key_type>
__global__ void cal_kv_intra_group_ind(key_type *keys,
                                       u_int32_t *indicator, 
                                       size_t kv_num,
                                       u_int32_t *par_rec_num, 
                                       size_t P);

template<typename key_type, typename val_type>
__global__ void kv_inter_group_insert(key_type *keys,
                                      val_type *vals,
                                      key_type *partition_keys,
                                      val_type *partition_vals,
                                      u_int32_t *indicator, 
                                      size_t kv_num,
                                      u_int32_t *par_rec_num_scan, 
                                      size_t P);