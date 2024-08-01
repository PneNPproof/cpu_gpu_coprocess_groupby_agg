#pragma once

#include "util.cuh"
#include <vector>
#include "par_result.cuh"

void cpu_groupby_agg_intra_partition_thread(std::vector<par_result> &par_result_vec,
                                            size_t par_num,
                                            key_type *host_groupby_keys_result,
                                            val_type *host_agg_vals_result,
                                            std::vector<size_t> &par_kv_begin,
                                            std::vector<size_t> &par_result_kv_num);