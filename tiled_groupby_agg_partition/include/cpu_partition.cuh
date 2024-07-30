#pragma once

#include "util.cuh"
#include "BS_thread_pool.hpp"
#include "par_result.cuh"

void cpu_task_assign_thread(key_type *host_keys,
                            val_type *host_vals,
                            size_t tile_num,
                            size_t tile_len,
                            size_t N,
                            u_int32_t *thread_local_par_rec_num,
                            u_int32_t *global_par_rec_num,
                            size_t cpu_partition_thread_num,
                            size_t P,
                            key_type *last_host_key_buffer,
                            val_type *last_host_val_buffer,
                            u_int32_t *hf_val_buffer,
                            u_int32_t *collect_loc_buffer,
                            size_t task_num,
                            BS::thread_pool &update_par_result_pool,
                            std::vector<par_result> &par_result_vec
                            );