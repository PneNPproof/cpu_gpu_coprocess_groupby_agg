#include "util.cuh"
#include "par_result.cuh"
#include "BS_thread_pool.hpp"
#include "cpu_partition.cuh"
#include <mutex>
#include <thread>
#include <cuco/detail/hash_functions/murmurhash3.cuh>

extern std::mutex g_counter_mutex;
extern std::mutex g_pool_mutex;
extern size_t g_counter;



// a single-thread task which cal hf(key) % P, update collect_loc_buffer and local_par_rec_num
void cal_thread_local_collect_ind_task(key_type *key_buffer,
                                       u_int32_t *hf_val_buffer,
                                       u_int32_t *collect_loc_buffer,
                                       size_t buffer_len,
                                       u_int32_t *local_par_rec_num,
                                       size_t P
                                       )
{
  // murmur3_32bit hf(0);
  cuco::detail::MurmurHash3_32<key_type> hf;

  for (size_t i = 0; i < buffer_len; i++)
  {
    auto key = key_buffer[i];
    auto hf_val = hf(key) % P;
    collect_loc_buffer[i] = local_par_rec_num[hf_val]++;
    hf_val_buffer[i] = hf_val;
  }
}

void cal_global_intra_par_ind_task(size_t par_bengin_ind,
                                   size_t par_end_ind,
                                   size_t task_num,
                                   size_t P,
                                   u_int32_t *all_local_par_rec_num,
                                   u_int32_t *global_par_rec_num
                                   )
{
  for (size_t i = 0; i < task_num; i++)
  {
    auto local_par_rec_num = all_local_par_rec_num + i * P;
    for (size_t j = par_bengin_ind; j < par_end_ind; j++)
    {
      auto temp = local_par_rec_num[j];
      local_par_rec_num[j] = global_par_rec_num[j];
      global_par_rec_num[j] += temp;
    }
  }
}

void cal_global_inter_par_collect_ind_and_collect_task(key_type *key_buffer,
                                                       val_type *val_buffer,
                                                       u_int32_t *hf_val_buffer,
                                                       u_int32_t *collect_loc_buffer,
                                                       size_t buffer_len,
                                                       u_int32_t *local_par_rec_num,
                                                       u_int32_t *global_par_rec_num,
                                                       key_type *last_host_key_buffer,
                                                       val_type *last_host_val_buffer
                                                       )
{
  for (size_t i = 0; i < buffer_len; i++)
  {
    auto par_ind = hf_val_buffer[i];
    auto collect_loc = collect_loc_buffer[i] + local_par_rec_num[par_ind] + global_par_rec_num[par_ind];
    // if (g_counter == 100) {
    //   printf("i %d hello\n", i);
    // }
    // if (g_counter == 100) {
    //   printf("val_buffer %ld\n", val_buffer[i]);
    // }
    last_host_key_buffer[collect_loc] = key_buffer[i];
    last_host_val_buffer[collect_loc] = val_buffer[i];
  }
}


// global_par_rec_num len P + 1
void cpu_task_assign_thread(key_type *host_keys,
                            val_type *host_vals,
                            size_t tile_num,
                            size_t tile_len,
                            size_t N,
                            u_int32_t *thread_local_par_rec_num,
                            u_int32_t *global_par_rec_num_all_tile,
                            size_t cpu_partition_thread_num,
                            size_t P,
                            key_type *last_host_key_buffer,
                            val_type *last_host_val_buffer,
                            u_int32_t *hf_val_buffer,
                            u_int32_t *collect_loc_buffer,
                            size_t task_num,
                            BS::thread_pool &update_par_result_pool,
                            std::vector<par_result> &par_result_vec
                            )
{ 
  BS::thread_pool cpu_partition_pool(cpu_partition_thread_num);
  while (1)
  {
    size_t tile_id;
    g_counter_mutex.lock();
    tile_id = g_counter++;
    g_counter_mutex.unlock();
    if (tile_id >= tile_num) {
      break;
    }
    auto tile_begin = tile_id * tile_len;
    auto tile_end = (tile_id + 1) * tile_len;
    if (tile_end > N) 
    {
      tile_end = N;
    }
    auto tile_kv_num = tile_end - tile_begin;
    auto host_key_buffer = host_keys + tile_begin;
    auto host_val_buffer = host_vals + tile_begin;
    
    // initialize some ds
    memset(thread_local_par_rec_num, 0x00, sizeof(u_int32_t) * P * task_num);
    // memset(global_par_rec_num, 0x00, sizeof(u_int32_t) * (P + 1));
    auto global_par_rec_num = global_par_rec_num_all_tile + tile_id * (P + 1);
    //

    // printf("tile id %ld\n", tile_id);

    // assign cal_thread_local_collect_ind_task 
    for (size_t task_ind = 0; task_ind < task_num; task_ind++)
    {
      auto task_kv_begin_ind = (task_ind * tile_kv_num) / task_num;
      auto task_kv_end_ind = ((task_ind + 1) * tile_kv_num) / task_num;
      auto task_buffer_len = task_kv_end_ind - task_kv_begin_ind;
      auto local_par_rec_num = thread_local_par_rec_num + P * task_ind;
      cpu_partition_pool.push_task(cal_thread_local_collect_ind_task,
                                   host_key_buffer + task_kv_begin_ind,
                                   hf_val_buffer + task_kv_begin_ind,
                                   collect_loc_buffer + task_kv_begin_ind,
                                   task_buffer_len,
                                   local_par_rec_num,
                                   P);
    }
    //

    cpu_partition_pool.wait_for_tasks();

    // assign cal_global_intra_par_ind_task
    for (size_t task_ind = 0; task_ind < task_num; task_ind++)
    {
      auto task_par_begin_ind = (task_ind * P) / task_num;
      auto task_par_end_ind = ((task_ind + 1) * P) / task_num;
      cpu_partition_pool.push_task(cal_global_intra_par_ind_task,
                                   task_par_begin_ind,
                                   task_par_end_ind,
                                   task_num,
                                   P,
                                   thread_local_par_rec_num,
                                   global_par_rec_num);
    }
    //

    cpu_partition_pool.wait_for_tasks();

    // exclusive prefix sum of global par rec num
    u_int32_t last_num = global_par_rec_num[0];
    global_par_rec_num[0] = 0;
    for (size_t i = 1; i < P + 1; i++)
    {
      auto now_num =global_par_rec_num[i];
      global_par_rec_num[i] = global_par_rec_num[i - 1] + last_num;
      last_num = now_num;
    }
    //

    // if (tile_id == 99)
    // {
    //   for (int i=0;i<tile_len;i++)
    //   {
    //     printf("last_host_key_buffer: %d %d \n", i, last_host_key_buffer[i]);
    //   }
    // }

    // assign cal_global_inter_par_collect_ind_and_collect_task
    for (size_t task_ind = 0; task_ind < task_num; task_ind++)
    {
      auto task_kv_begin_ind = (task_ind * tile_kv_num) / task_num;
      auto task_kv_end_ind = ((task_ind + 1) * tile_kv_num) / task_num;
      auto task_buffer_len = task_kv_end_ind - task_kv_begin_ind;
      auto local_par_rec_num = thread_local_par_rec_num + P * task_ind;

      cpu_partition_pool.push_task(cal_global_inter_par_collect_ind_and_collect_task,
                                   host_key_buffer + task_kv_begin_ind,
                                   host_val_buffer + task_kv_begin_ind,
                                   hf_val_buffer + task_kv_begin_ind,
                                   collect_loc_buffer + task_kv_begin_ind,
                                   task_buffer_len,
                                   local_par_rec_num,
                                   global_par_rec_num,
                                   last_host_key_buffer,
                                   last_host_val_buffer);
    }
    //

    cpu_partition_pool.wait_for_tasks();

    // update par_result
    g_pool_mutex.lock();
    update_par_result_pool.push_task(update_par_result_task, last_host_key_buffer, last_host_val_buffer, global_par_rec_num, P, std::ref(par_result_vec));
    g_pool_mutex.unlock();
    //

    // now host_kv_buffer has been collect into last_host_kv_buffer, let host_kv_buffer be new last_host_kv_buffer
    last_host_key_buffer = host_key_buffer;
    last_host_val_buffer = host_val_buffer;
    //
  }


}