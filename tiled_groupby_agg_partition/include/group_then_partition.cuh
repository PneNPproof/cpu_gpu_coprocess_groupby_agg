#pragma once
#include "util.cuh"
#include "par_result.cuh"

template<typename key_type, typename val_type>
void groupby_agg_partition(key_type *host_keys_buffer,
                           val_type *host_vals_buffer,
                           size_t kv_buffer_len,
                           size_t tile_len,
                           size_t P,
                           std::vector<par_result> &par_result_vec,
                           size_t nstreams);