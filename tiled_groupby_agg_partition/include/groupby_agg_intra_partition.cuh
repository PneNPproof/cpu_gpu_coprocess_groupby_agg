#pragma once

#include "par_result.cuh"

void groupby_agg_intra_partition(std::vector<par_result> &par_result_vec,
                                      key_type *&host_groupby_keys_result,
                                      val_type *&host_agg_vals_result,
                                      size_t Capacity,
                                      size_t min_load_num,
                                      size_t max_load_num,
                                      size_t nstreams,
                                      std::vector<size_t> &par_kv_begin,
                                      std::vector<size_t> &par_result_kv_num);