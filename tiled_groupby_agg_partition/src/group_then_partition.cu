#include <thread>
#include <mutex>
#include "group_then_partition.cuh"
#include "kernel.cuh"
#include "BS_thread_pool.hpp"
#include "util.cuh"
#include "cpu_partition.cuh"

#define UPDATE_PAR_RESULT_THREAD_NUM 4

std::mutex g_counter_mutex;
std::mutex g_pool_mutex;
size_t g_counter;


u_int32_t* copy_to_host_display(u_int32_t *device_arr, size_t len, cudaStream_t stream)
{
  u_int32_t *host_arr;
  host_arr = (u_int32_t *)malloc(sizeof(u_int32_t)*len);
  cudaMemcpyAsync(host_arr, device_arr, sizeof(u_int32_t)*len, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return host_arr;
}


template<typename key_type, typename val_type>
void compute_groupby_agg_result(key_type *device_groupby_keys,
                                val_type *device_agg_vals,
                                key_type *device_ht_keys,
                                val_type *device_ht_vals,
                                u_int32_t *indicator,
                                size_t device_kv_num,
                                size_t device_ht_sz,
                                key_type empty_key,
                                key_type *host_collect_sz_2,
                                u_int32_t *host_kv_num_4,
                                u_int32_t *host_collect_sz_5,
                                key_type* &device_result_keys,
                                val_type* &device_result_vals,
                                size_t &device_result_num,
                                cudaStream_t stream,
                                void *temp_store,
                                size_t temp_store_bytes)
{
#ifdef MEASURE_TIME
  RuntimeMeasurement timer;
#endif

  #ifdef MEASURE_TIME
  timer.start();
  #endif

  // prepare first kernel: device array initialization and kernel configuration
  auto groupby_keys_1 = device_groupby_keys;
  auto agg_vals_1 = device_agg_vals;
  auto ht_keys_1 = device_ht_keys;
  auto ht_vals_1 = device_ht_vals;
  auto kv_num_1 = device_kv_num;
  auto one_third_ht_sz_1 = device_ht_sz / 3;
  auto ht_sz_1 = device_ht_sz;
  auto kv_indicator_1 = indicator;
  auto ht_indicator_1 = device_ht_vals + one_third_ht_sz_1;
  auto empty_key_1 = empty_key;
  auto random_seed_1 = std::rand();
  size_t bdx_1 = 256;
  size_t gdx_1 = (kv_num_1 + bdx_1 - 1) / bdx_1;

  // launch kernel 1
  hash_into_one_third_ht<key_type, val_type><<<gdx_1, bdx_1, 0, stream>>>(groupby_keys_1,
                                                                          agg_vals_1,
                                                                          ht_keys_1,
                                                                          ht_vals_1,
                                                                          kv_num_1,
                                                                          one_third_ht_sz_1,
                                                                          ht_sz_1,
                                                                          kv_indicator_1,
                                                                          ht_indicator_1,
                                                                          empty_key_1,
                                                                          random_seed_1);


  #ifdef MEASURE_TIME
  cudaStreamSynchronize(stream);
  timer.stop();
  timer.print_elapsed_time("kernel 1");
  #endif

  #ifdef MEASURE_TIME
  timer.start();
  #endif

  // prepare second kernel: device array initialization and kernel configuration
  auto indicator_2 = ht_indicator_1;
  auto indicator_scan_2 = device_ht_keys + one_third_ht_sz_1;
  auto n_2 = one_third_ht_sz_1;


  // auto host_ptr = copy_to_host_display(indicator_2, n_2, stream);


  scan_inclusive<key_type, val_type>(indicator_2,
                               indicator_scan_2,
                               n_2,
                               stream,
                               temp_store,
                               temp_store_bytes);
  auto &collect_sz_2 = (*host_collect_sz_2);
  // cudaMemcpy(&collect_sz_2, &indicator_scan_2[n_2 - 1], sizeof(key_type), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(&collect_sz_2, &indicator_scan_2[n_2 - 1], sizeof(key_type), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);


  

  
  auto ht_keys_2 = ht_keys_1;
  auto ht_vals_2 = ht_vals_1;
  auto collect_keys_2 = device_ht_keys + (device_ht_sz - collect_sz_2);
  auto collect_vals_2 = device_ht_vals + (device_ht_sz - collect_sz_2);
  auto one_third_ht_sz_2 = one_third_ht_sz_1;
  auto ht_indicator_scan_2 = indicator_scan_2;
  auto empty_key_2 = empty_key;
  size_t bdx_2 = 256;
  size_t gdx_2 = (one_third_ht_sz_2 + bdx_2 - 1) / bdx_2;

  // launch kernel 2
  collect_one_third_ht_result<key_type, val_type><<<gdx_2, bdx_2, 0, stream>>>(ht_keys_2,
                                                                               ht_vals_2,
                                                                               collect_keys_2,
                                                                               collect_vals_2,
                                                                               one_third_ht_sz_2,
                                                                               ht_indicator_scan_2,
                                                                               empty_key_2);
  
  
  #ifdef MEASURE_TIME
  cudaStreamSynchronize(stream);
  timer.stop();
  timer.print_elapsed_time("kernel 2");
  #endif


  #ifdef MEASURE_TIME
  timer.start();
  #endif

  // prepare third kernel: device array initialization and kernel configuration
  auto indicator_3 = kv_indicator_1;
  auto indicator_scan_3 = indicator_3;
  auto n_3 = device_ht_sz;
  scan_inclusive<key_type, val_type>(indicator_3,
                               indicator_scan_3,
                               n_3,
                               stream,
                               temp_store,
                               temp_store_bytes);

  auto groupby_keys_3 = device_groupby_keys;
  auto agg_vals_3 = device_agg_vals;
  auto ht_keys_3 = device_ht_keys;
  auto ht_vals_3 = device_ht_vals;
  auto kv_indicator_scan_3 = indicator_scan_3;
  auto kv_num_3 = device_kv_num;
  size_t bdx_3 = 256;
  size_t gdx_3 = (kv_num_3 + bdx_3 - 1) / bdx_3;

  // launch kernel 3
  collect_unhashed_kv_into_ht<key_type, val_type><<<gdx_3, bdx_3, 0, stream>>>(groupby_keys_3,
                                                                               agg_vals_3,
                                                                               ht_keys_3,
                                                                               ht_vals_3,
                                                                               kv_indicator_scan_3,
                                                                               kv_num_3);

  

  #ifdef MEASURE_TIME
  cudaStreamSynchronize(stream);
  timer.stop();
  timer.print_elapsed_time("kernel 3");
  #endif


  #ifdef MEASURE_TIME
  timer.start();
  #endif
  // prepare fourth kernel: device array initialization and kernel configuration
  auto groupby_keys_4 = device_ht_keys;
  auto agg_vals_4 = device_ht_vals;
  auto ht_keys_4 = device_groupby_keys;
  auto ht_vals_4 = device_agg_vals;
  auto &kv_num_4 = (*host_kv_num_4);
  // cudaMemcpy(&kv_num_4, &indicator_scan_3[n_3 - 1], sizeof(u_int32_t), cudaMemcpyDeviceToHost);
  cudaMemcpyAsync(&kv_num_4, &indicator_scan_3[n_3 - 1], sizeof(u_int32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  

  auto ht_sz_4 = device_kv_num;
  auto ht_indicator_4 = indicator;
  auto empty_key_4 = empty_key;
  auto random_seed_4 = std::rand();
  // cudaMemset(ht_keys_4, 0xff, sizeof(key_type) * ht_sz_4);
  cudaMemsetAsync(ht_keys_4, 0xff, sizeof(key_type) * ht_sz_4, stream);
  // cudaMemset(ht_vals_4, 0x00, sizeof(val_type) * ht_sz_4);
  // cudaMemset(ht_indicator_4, 0x00, sizeof(u_int32_t) * ht_sz_4);
  cudaMemsetAsync(ht_vals_4, 0x00, sizeof(val_type) * ht_sz_4, stream);
  cudaMemsetAsync(ht_indicator_4, 0x00, sizeof(u_int32_t) * ht_sz_4, stream);
  size_t bdx_4 = 256;
  size_t gdx_4 = (kv_num_4 + bdx_4 - 1) / bdx_4;

  // launch kernel 4
  kv_hash_agg_into_ht<key_type, val_type><<<gdx_4, bdx_4, 0, stream>>>(groupby_keys_4,
                                                                       agg_vals_4,
                                                                       ht_keys_4,
                                                                       ht_vals_4,
                                                                       kv_num_4,
                                                                       ht_sz_4,
                                                                       ht_indicator_4,
                                                                       empty_key_4,
                                                                       random_seed_4);


  #ifdef MEASURE_TIME
  cudaStreamSynchronize(stream);
  timer.stop();
  timer.print_elapsed_time("kernel 4");
  #endif


  #ifdef MEASURE_TIME
  timer.start();
  #endif
  // prepare fifth kernel: device array initialization and kernel configuration
  auto ht_keys_5 = device_groupby_keys;
  auto ht_vals_5 = device_agg_vals;
  auto indicator_5 = ht_indicator_4;
  auto indicator_scan_5 = indicator_5;
  auto n_5 = device_ht_sz;
  scan_inclusive<key_type, val_type>(indicator_5,
                                     indicator_scan_5,
                                     n_5,
                                     stream,
                                     temp_store,
                                     temp_store_bytes);
  auto &collect_sz_5 = (*host_collect_sz_5);
  cudaMemcpyAsync(&collect_sz_5, &indicator_scan_5[n_5 - 1], sizeof(u_int32_t), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  auto collect_keys_5 = device_ht_keys + (device_ht_sz - collect_sz_2 - collect_sz_5);
  auto collect_vals_5 = device_ht_vals + (device_ht_sz - collect_sz_2 - collect_sz_5);
  auto ht_sz_5 = device_kv_num;
  auto ht_indicator_scan_5 = indicator_scan_5;
  auto empty_key_5 = empty_key;
  size_t bdx_5 = 256;
  size_t gdx_5 = (ht_sz_5 + bdx_5 - 1) / bdx_5;

  // launch kernel 5
  combine_ht_result_with_previous_result<key_type, val_type><<<gdx_5, bdx_5, 0, stream>>>(ht_keys_5,
                                                                                          ht_vals_5,
                                                                                          collect_keys_5,
                                                                                          collect_vals_5,
                                                                                          ht_sz_5,
                                                                                          ht_indicator_scan_5,
                                                                                          empty_key_5);

  #ifdef MEASURE_TIME
  cudaStreamSynchronize(stream);
  timer.stop();
  timer.print_elapsed_time("kernel 5");
  #endif

  device_result_keys = collect_keys_5;
  device_result_vals = collect_vals_5;
  device_result_num = collect_sz_2 + collect_sz_5;
}

template<typename key_type, typename val_type>
void compute_partition_result(key_type *keys,
                              val_type *vals,
                              u_int32_t *indicator,
                              size_t kv_num,
                              u_int32_t *par_rec_num,
                              size_t P,
                              key_type *partition_keys,
                              val_type *partition_vals,
                              cudaStream_t stream,
                              void *temp_store,
                              size_t temp_store_bytes)
{
  #ifdef MEASURE_TIME
  RuntimeMeasurement timer;
  #endif
  #ifdef MEASURE_TIME
  timer.start();
  #endif
  // calculate kv intra group ind
  auto keys_1 = keys;
  auto indicator_1 = indicator;
  auto kv_num_1 = kv_num;
  auto par_rec_num_1 = par_rec_num;
  auto P_1 = P;
  size_t bdx_1 = 256;
  size_t gdx_1 = (kv_num_1 + bdx_1 - 1) / bdx_1;
  cal_kv_intra_group_ind<key_type><<<gdx_1, bdx_1, 0, stream>>>(keys_1,
                                                                indicator_1,
                                                                kv_num_1,
                                                                par_rec_num_1,
                                                                P_1);

  //

  // scan par_rec_num
  scan_exclusive(par_rec_num_1,
                 par_rec_num_1,
                 P,
                 stream,
                 temp_store,
                 temp_store_bytes);
  //
  #ifdef MEASURE_TIME
  cudaStreamSynchronize(stream);
  timer.stop();
  timer.print_elapsed_time("kernel 6");
  #endif

  #ifdef MEASURE_TIME
  timer.start();
  #endif
  // calculate kv inter group ind and insert
  auto keys_2 = keys_1;
  auto vals_2 = vals;
  auto partition_keys_2 = partition_keys;
  auto partition_vals_2 = partition_vals;
  auto indicator_2 = indicator_1;
  auto kv_num_2 = kv_num_1;
  auto par_rec_num_scan_2 = par_rec_num_1;
  auto P_2 = P_1;
  auto bdx_2 = 256;
  auto gdx_2 = (kv_num_2 + bdx_2 - 1) / bdx_2;
  kv_inter_group_insert<key_type, val_type><<<gdx_2, bdx_2, 0, stream>>>(keys_2,
                                                                         vals_2,
                                                                         partition_keys_2,
                                                                         partition_vals_2,
                                                                         indicator_2,
                                                                         kv_num_2,
                                                                         par_rec_num_scan_2,
                                                                         P_2);
  //
  #ifdef MEASURE_TIME
  cudaStreamSynchronize(stream);
  timer.stop();
  timer.print_elapsed_time("kernel 7");
  #endif
}

template<typename key_type, typename val_type>
void groupby_agg_partition_thread(key_type *device_groupby_keys,
                                  val_type *device_agg_vals,
                                  key_type *device_ht_keys,
                                  val_type *device_ht_vals,
                                  u_int32_t *indicator,
                                  size_t tile_num,
                                  size_t tile_len,
                                  key_type empty_key,
                                  key_type *host_collect_sz_2,
                                  u_int32_t *host_kv_num_4,
                                  u_int32_t *host_collect_sz_5,
                                  cudaStream_t stream,
                                  void *temp_store,
                                  size_t temp_store_bytes,
                                  u_int32_t *par_rec_num,
                                  size_t P,
                                  BS::thread_pool &update_par_result_pool,
                                  key_type *host_keys,
                                  val_type *host_vals,
                                  u_int32_t *host_par_pos,
                                  size_t N,
                                  std::vector<par_result> &par_result_vec)
{
  #ifdef MEASURE_TIME
  RuntimeMeasurement timer;
  #endif

  while (1) {
    size_t tile_id;
    g_counter_mutex.lock();
    tile_id = g_counter++;
    g_counter_mutex.unlock();
    if (tile_id >= tile_num) {
      break;
    }
    
    // initialize data 
    auto tile_begin = tile_id * tile_len;
    auto tile_end = (tile_id + 1) * tile_len;
    if (tile_end > N) {
      tile_end = N;
    }
    auto device_kv_num = tile_end - tile_begin;
    auto device_ht_sz = device_kv_num;

    auto tile_keys_sz = sizeof(key_type) * device_kv_num;
    auto tile_vals_sz = sizeof(val_type) * device_kv_num;

    #ifdef MEASURE_TIME
    timer.start();
    #endif

    cudaMemcpyAsync(device_groupby_keys, host_keys + tile_begin, tile_keys_sz, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(device_agg_vals, host_vals + tile_begin, tile_vals_sz, cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(device_ht_keys, 0xff, sizeof(key_type) * device_ht_sz, stream);
    cudaMemsetAsync(device_ht_vals, 0x00, sizeof(val_type) * device_ht_sz, stream);
    cudaMemsetAsync(indicator, 0x00, sizeof(u_int32_t) * device_ht_sz, stream);
    cudaMemsetAsync(par_rec_num, 0x00, sizeof(u_int32_t) * P, stream);

    #ifdef MEASURE_TIME
    cudaStreamSynchronize(stream);
    timer.stop();
    timer.print_elapsed_time("groupby_agg_partition copy in");
    #endif
    //

    // gorupby_agg phase
    key_type *device_result_keys;
    val_type *device_result_vals;
    size_t device_result_num;

    compute_groupby_agg_result<key_type, val_type>(device_groupby_keys,
                                                   device_agg_vals,
                                                   device_ht_keys,
                                                   device_ht_vals,
                                                   indicator,
                                                   device_kv_num,
                                                   device_ht_sz,
                                                   empty_key,
                                                   host_collect_sz_2,
                                                   host_kv_num_4,
                                                   host_collect_sz_5,
                                                   device_result_keys,
                                                   device_result_vals,
                                                   device_result_num,
                                                   stream,
                                                   temp_store,
                                                   temp_store_bytes);
    //


    cudaStreamSynchronize(stream);



    // partition phase
    compute_partition_result<key_type, val_type>(device_result_keys,
                                                 device_result_vals,
                                                 indicator,
                                                 device_result_num,
                                                 par_rec_num,
                                                 P,
                                                 device_groupby_keys,
                                                 device_agg_vals,
                                                 stream,
                                                 temp_store,
                                                 temp_store_bytes);
    //

    #ifdef MEASURE_TIME
    timer.start();
    #endif

    // copy par_pos from device to host
    auto par_pos_tile_len = (P + 1);
    auto par_pos_tile_begin = tile_id * par_pos_tile_len;
    cudaMemcpyAsync(host_par_pos + par_pos_tile_begin, par_rec_num, P * sizeof(u_int32_t), cudaMemcpyDeviceToHost, stream);
    (*(host_par_pos + par_pos_tile_begin + P)) = device_result_num;
    cudaStreamSynchronize(stream);
    //

    // copy par_result from device to host
    auto tile_par_pos = host_par_pos + par_pos_tile_begin;
    auto tile_key_buffer = host_keys + tile_begin;
    auto tile_val_buffer = host_vals + tile_begin;
    cudaMemcpyAsync(tile_key_buffer, device_groupby_keys, device_result_num * sizeof(key_type), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(tile_val_buffer, device_agg_vals, device_result_num * sizeof(val_type), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    //

    #ifdef MEASURE_TIME
    cudaStreamSynchronize(stream);
    timer.stop();
    timer.print_elapsed_time("groupby_agg_partition copy out");
    #endif

    g_pool_mutex.lock();
    update_par_result_pool.push_task(update_par_result_task, tile_key_buffer, tile_val_buffer, tile_par_pos, P, std::ref(par_result_vec));
    g_pool_mutex.unlock();
  }
}

template<typename key_type, typename val_type>
void groupby_agg_partition(key_type *host_keys_buffer,
                           val_type *host_vals_buffer,
                           size_t kv_buffer_len,
                           size_t tile_len,
                           size_t P,
                           std::vector<par_result> &par_result_vec,
                           size_t nstreams)
{
  auto tile_num = (kv_buffer_len + tile_len - 1) / tile_len;
  g_counter = 0;

  BS::thread_pool update_par_result_pool(UPDATE_PAR_RESULT_THREAD_NUM);

  // allocate host pinned memory
  u_int32_t *host_par_pos;
  key_type *host_collect_sz_2;
  u_int32_t *host_kv_num_4;
  u_int32_t *host_collect_sz_5;
  cudaMallocHost(&host_par_pos, sizeof(u_int32_t) * (P + 1) * tile_num);
  cudaMallocHost(&host_collect_sz_2, sizeof(key_type) * nstreams);
  cudaMallocHost(&host_kv_num_4, sizeof(u_int32_t) * nstreams);
  cudaMallocHost(&host_collect_sz_5, sizeof(u_int32_t) * nstreams);

  /// allocate for cpu partition
  key_type *last_host_tile_key_buffer;
  val_type *last_host_tile_val_buffer;
  cudaMallocHost(&last_host_tile_key_buffer, sizeof(key_type) * tile_len);
  cudaMallocHost(&last_host_tile_val_buffer, sizeof(val_type) * tile_len);

  size_t cpu_partition_thread_num = 12;
  size_t task_num = 12;
  u_int32_t *thread_local_par_rec_num;
  u_int32_t *global_par_rec_num_all_tile;
  u_int32_t *hf_val_buffer;
  u_int32_t *collect_loc_buffer;
  cudaMallocHost(&thread_local_par_rec_num, sizeof(u_int32_t) * P * task_num);
  cudaMallocHost(&global_par_rec_num_all_tile, sizeof(u_int32_t) * (P + 1) * tile_num);
  cudaMallocHost(&hf_val_buffer, sizeof(u_int32_t) * tile_len);
  cudaMallocHost(&collect_loc_buffer, sizeof(u_int32_t) * tile_len);
  memset(global_par_rec_num_all_tile, 0x00, sizeof(u_int32_t) * (P + 1) * tile_num);
  ///


  /// allocate device memory
  key_type **groupby_keys = (key_type **)malloc(sizeof(key_type *) * nstreams);
  val_type **agg_vals = (val_type **)malloc(sizeof(val_type *) * nstreams);
  key_type **ht_keys = (key_type **)malloc(sizeof(key_type *) * nstreams);
  val_type **ht_vals = (val_type **)malloc(sizeof(val_type *) * nstreams);
  u_int32_t **indicator = (u_int32_t **)malloc(sizeof(u_int32_t *) * nstreams);
  u_int32_t **par_rec_num = (u_int32_t **)malloc(sizeof(u_int32_t *) * nstreams);

  void **temp_store = (void **)malloc(sizeof(void *) * nstreams);
  size_t temp_store_bytes = sizeof(key_type) * tile_len;

  auto dev_ptr = pre_device_alloc(groupby_keys, agg_vals, ht_keys, ht_vals, indicator, par_rec_num, temp_store, tile_len, P, nstreams);
  ///

  // create nstreams threads to deal with tile_num tiles, each thread is bound to a cuda stream
  cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
  for (size_t i = 0; i < nstreams; i++) {
    cudaStreamCreateWithFlags(streams + i, cudaStreamNonBlocking);
  }


  RuntimeMeasurement timer_phase1;
  timer_phase1.start();


  std::vector<std::thread> nthreads(nstreams);
  for (size_t i = 0; i < nstreams; i++)
  {
    nthreads[i] = std::thread(groupby_agg_partition_thread<key_type, val_type>,
                              groupby_keys[i],
                              agg_vals[i],
                              ht_keys[i],
                              ht_vals[i],
                              indicator[i],
                              tile_num,
                              tile_len,
                              empty_k,
                              host_collect_sz_2 + i,
                              host_kv_num_4 + i,
                              host_collect_sz_5 +i,
                              streams[i],
                              temp_store[i],
                              temp_store_bytes,
                              par_rec_num[i],
                              P,
                              std::ref(update_par_result_pool),
                              host_keys_buffer,
                              host_vals_buffer,
                              host_par_pos,
                              kv_buffer_len,
                              std::ref(par_result_vec));
  }
  //


  /// cpu coprocess deal with tiles
  std::thread cpu_assign_thread;
  cpu_assign_thread = std::thread(cpu_task_assign_thread,
                                  host_keys_buffer,
                                  host_vals_buffer,
                                  tile_num,
                                  tile_len,
                                  kv_buffer_len,
                                  thread_local_par_rec_num,
                                  global_par_rec_num_all_tile,
                                  cpu_partition_thread_num,
                                  P,
                                  last_host_tile_key_buffer,
                                  last_host_tile_val_buffer,
                                  hf_val_buffer,
                                  collect_loc_buffer,
                                  task_num,
                                  std::ref(update_par_result_pool),
                                  std::ref(par_result_vec));
  ///

  // wait all task finished
  cpu_assign_thread.join();
  for (size_t i = 0; i < nstreams; i++) 
  {
    nthreads[i].join();
  }
  
  update_par_result_pool.wait_for_tasks();
  //

  timer_phase1.stop();
  timer_phase1.print_elapsed_time("phase 1");

  // free phase
  for (size_t i = 0; i < nstreams; i++) {
    cudaStreamDestroy(streams[i]);
  }

  cudaFree(dev_ptr);
  cudaFreeHost(host_par_pos);
  cudaFreeHost(host_collect_sz_2);
  cudaFreeHost(host_kv_num_4);
  cudaFreeHost(host_collect_sz_5);


  // cudaFreeHost(hf_val_buffer);
  // cudaFreeHost(collect_loc_buffer);
  // cudaFreeHost(thread_local_par_rec_num);
  // cudaFreeHost(global_par_rec_num_all_tile);


  free(groupby_keys);
  free(agg_vals);
  free(ht_keys);
  free(ht_vals);
  free(indicator);
  free(par_rec_num);
  free(streams);

}


template
void groupby_agg_partition(key_type *host_keys_buffer,
                           val_type *host_vals_buffer,
                           size_t kv_buffer_len,
                           size_t tile_len,
                           size_t P,
                           std::vector<par_result> &par_result_vec,
                           size_t nstreams);
