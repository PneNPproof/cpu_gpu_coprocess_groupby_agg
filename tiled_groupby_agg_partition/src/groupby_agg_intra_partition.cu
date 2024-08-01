#include "groupby_agg_intra_partition.cuh"
#include <queue>
#include <vector>
#include <cub/cub.cuh>
#include <thread>
#include <mutex>

#include "kernel.cuh"
#include "util.cuh"
#include "cpu_groupby_agg_intra_partition.cuh"

size_t g_par_counter;
std::mutex g_par_counter_mutex;

static std::vector<par_result> merge_partition(std::vector<par_result> &par_result_vec, size_t K) {
  auto cmp = [](par_result &left, par_result &right) { return left.size > right.size; };
  std::priority_queue H(par_result_vec.begin(), par_result_vec.end(), cmp);

  while (1) {
    if (H.size()==1) {
      break;
    }
    auto tmp_par = H.top();
    H.pop();
    auto top_par = H.top();
    if (tmp_par.size + top_par.size > K) {
      H.push(tmp_par);
      break;
    }
    do {
      tmp_par.merge(top_par);
      H.pop();
      if (H.size() == 0) {
        break;
      }
      top_par = H.top();
    } while (top_par.size + tmp_par.size <= K);
    H.push(tmp_par);
  }

  std::vector<par_result> rv;

  auto len = H.size();
  for (size_t i=0;i<len;i++) {
    rv.push_back(H.top());
    H.pop();
  }
  
  return rv;
}


void *pre_alloc_device_memory(key_type **groupby_keys,
                              val_type **agg_vals, 
                              key_type **ht_keys, 
                              val_type **ht_vals,
                              u_int32_t **indicator,
                              void **temp_storage,
                              u_int32_t **block_result_num,
                              size_t Capacity,
                              size_t max_load_num,
                              size_t temp_storage_bytes,
                              size_t max_thread_blocks_num, 
                              size_t nstreams) 
{
  auto total_size = ((sizeof(key_type) + sizeof(val_type)) * (max_load_num + Capacity) + sizeof(u_int32_t) * (Capacity + (max_thread_blocks_num + 1)) + temp_storage_bytes) * nstreams;
  void *dev_ptr;
  cudaMalloc(&dev_ptr, total_size);
  size_t offset = 0;
  for (int i=0; i<nstreams; i++) {
    groupby_keys[i] = static_cast<key_type *>(dev_ptr + offset);
    offset += max_load_num * sizeof(key_type);
    agg_vals[i] = static_cast<val_type *>(dev_ptr + offset);
    offset += max_load_num * sizeof(val_type);
    ht_keys[i] = static_cast<key_type *>(dev_ptr + offset);
    offset += Capacity * sizeof(key_type);
    ht_vals[i] = static_cast<val_type *>(dev_ptr + offset);
    offset += Capacity * sizeof(val_type);
    indicator[i] = static_cast<u_int32_t *>(dev_ptr + offset);
    offset += Capacity * sizeof(u_int32_t);
    temp_storage[i] = (dev_ptr + offset);
    offset += temp_storage_bytes;
    block_result_num[i] = static_cast<u_int32_t *>(dev_ptr + offset);
    offset += (max_thread_blocks_num + 1) * sizeof(u_int32_t);
  }

  return dev_ptr;
}


static void par_result_in_continous_mem(par_result &pr, key_type *kc_begin, val_type *vc_begin) {
  auto key_cp_dst = kc_begin;
  auto val_cp_dst = vc_begin;
  
  auto &key_seg = pr.key;
  auto &val_seg = pr.val;
  auto &sz_seg = pr.end_indicator;
  auto seg_num = key_seg.size();
  for (size_t i = 0; i<seg_num; i++) {
    memcpy(key_cp_dst, key_seg[i], sizeof(key_type) * sz_seg[i]);
    memcpy(val_cp_dst, val_seg[i], sizeof(val_type) * sz_seg[i]);
    key_cp_dst += sz_seg[i];
    val_cp_dst += sz_seg[i];
  }

  par_result new_pr;
  new_pr.push(kc_begin, vc_begin, pr.size);
  pr = new_pr;
}


u_int32_t* _copy_to_host_display(u_int32_t *device_arr, size_t len, cudaStream_t stream)
{
  u_int32_t *host_arr;
  host_arr = (u_int32_t *)malloc(sizeof(u_int32_t)*len);
  cudaMemcpyAsync(host_arr, device_arr, sizeof(u_int32_t)*len, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  return host_arr;
}

void groupby_agg_and_statistic(key_type *groupby_keys, 
                               val_type *agg_vals, 
                               key_type *ht_keys, 
                               val_type *ht_vals,
                               u_int32_t *indicator, 
                               u_int32_t *block_result_num, 
                               u_int32_t *real_time_result_num, 
                               u_int32_t kv_num, 
                               u_int32_t Capacity, 
                               cudaStream_t stream,
                               key_type empty_key) 
{

  #ifdef MEASURE_TIME_2
  RuntimeMeasurement timer;
  #endif
  #ifdef MEASURE_TIME_2
  timer.start();
  #endif

  size_t bdx_1 = 256;
  size_t gdx_1 = (kv_num + bdx_1 - 1) / bdx_1;
  auto kernel_1_block_num = gdx_1;

  // one more elements for reduce result
  cudaMemsetAsync(block_result_num + kernel_1_block_num, 0x00, sizeof(u_int32_t), stream);
  //

  groupby_agg_and_statistic_kernel<<<gdx_1, bdx_1, 0, stream>>>
                                  (groupby_keys,
                                   agg_vals,
                                   ht_keys,
                                   ht_vals,
                                   indicator,
                                   block_result_num,
                                   kv_num,
                                   Capacity,
                                   empty_key);

  #ifdef MEASURE_TIME_2
  cudaStreamSynchronize(stream);
  timer.stop();
  timer.print_elapsed_time("groupby_agg_and_statistic_kernel");
  #endif


  #ifdef MEASURE_TIME_2
  timer.start();
  #endif

  size_t bdx_2 = 256;
  size_t gdx_2 = (kernel_1_block_num + 2 * bdx_2 -1) / (2 * bdx_2);
  auto reuduce_sum = block_result_num + kernel_1_block_num;
  device_reduce_kernel<<<gdx_2, bdx_2, 0, stream>>>(block_result_num, kernel_1_block_num, reuduce_sum);

  cudaMemcpyAsync(real_time_result_num, reuduce_sum, sizeof(u_int32_t), cudaMemcpyDeviceToHost, stream);

  #ifdef MEASURE_TIME_2
  cudaStreamSynchronize(stream);
  timer.stop();
  timer.print_elapsed_time("device_reduce_kernel");
  #endif
}


void groupby_agg_intra_partition_thread(std::vector<par_result> &par_result_vec, 
                                        key_type *groupby_keys, 
                                        val_type *agg_vals, 
                                        key_type *ht_keys, 
                                        val_type *ht_vals,
                                        u_int32_t *indicator,
                                        void *temp_storage,
                                        u_int32_t *block_result_num, 
                                        key_type *host_groupby_keys_result, 
                                        val_type *host_agg_vals_result, 
                                        size_t Capacity,
                                        size_t temp_storage_bytes, 
                                        size_t max_load_num, 
                                        size_t min_load_num,
                                        cudaStream_t stream, 
                                        u_int32_t *real_time_result_num, 
                                        size_t par_num, 
                                        std::vector<size_t> &par_kv_begin, 
                                        std::vector<size_t> &par_result_kv_num)
{
  #ifdef MEASURE_TIME_2
  RuntimeMeasurement timer;
  #endif

  size_t next_load_kv_num;
  size_t already_load_kv_num;

  // loop deal with tasks
  while (1)
  {
    // fetch a par task
    size_t par_ind;
    g_par_counter_mutex.lock();
    par_ind = g_par_counter++;
    g_par_counter_mutex.unlock();

    if (par_ind >= par_num)
    {
      break;
    }
    //

    #ifdef MEASURE_TIME_2
    timer.start();
    #endif

    // prepare ht_keys, ht_vals and indicator
    cudaMemsetAsync(ht_keys, 0xff, sizeof(key_type) * Capacity, stream);
    cudaMemsetAsync(ht_vals, 0x00, sizeof(val_type) * Capacity, stream);
    cudaMemsetAsync(indicator, 0x00, sizeof(u_int32_t) * Capacity, stream);
    //

    already_load_kv_num = 0;
    auto &pr = par_result_vec[par_ind];
    key_type empty_key = 0xffffffff;

    // copy host par kv into continus host mem
    par_result_in_continous_mem(pr, host_groupby_keys_result + par_kv_begin[par_ind], host_agg_vals_result + par_kv_begin[par_ind]);
    //

    #ifdef MEASURE_TIME_2
    cudaStreamSynchronize(stream);
    timer.stop();
    timer.print_elapsed_time("par_result_in_continous_mem");
    #endif

    // try to deal with all kv for this par in gpu
    auto rest_kv_num = pr.size;
    while (rest_kv_num > 0) {
      // ensure next load don't surpass max_load_num
      next_load_kv_num = max_load_num - already_load_kv_num;
      //
      if (rest_kv_num <= next_load_kv_num) {
        #ifdef MEASURE_TIME_2
        timer.start();
        #endif
        pr.pop2device(groupby_keys, agg_vals, rest_kv_num, stream);
        #ifdef MEASURE_TIME_2
        cudaStreamSynchronize(stream);
        timer.stop();
        timer.print_elapsed_time("pop2device");
        #endif
        groupby_agg_and_statistic(groupby_keys,
                                  agg_vals,
                                  ht_keys,
                                  ht_vals,
                                  indicator,
                                  block_result_num,
                                  real_time_result_num,
                                  rest_kv_num,
                                  Capacity,
                                  stream,
                                  empty_key);
        rest_kv_num = 0;
      }
      else {
        if (already_load_kv_num > min_load_num) {
          std::cout<<"hash table overflow\n";
          exit(2);
        }
        #ifdef MEASURE_TIME_2
        timer.start();
        #endif
        pr.pop2device(groupby_keys, agg_vals, next_load_kv_num, stream);
        #ifdef MEASURE_TIME_2
        cudaStreamSynchronize(stream);
        timer.stop();
        timer.print_elapsed_time("pop2device");
        #endif
        groupby_agg_and_statistic(groupby_keys,
                                  agg_vals,
                                  ht_keys,
                                  ht_vals,
                                  indicator,
                                  block_result_num,
                                  real_time_result_num,
                                  next_load_kv_num,
                                  Capacity,
                                  stream,
                                  empty_key);
        rest_kv_num -= next_load_kv_num;
      }
      cudaStreamSynchronize(stream);
      already_load_kv_num += real_time_result_num[0];
    }
    //

    // now all kv in this par has been groupby_agg into ht, we collect result

    #ifdef MEASURE_TIME_2
    timer.start();
    #endif

    // scan indicator
    scan_exclusive(indicator, indicator, Capacity, stream, temp_storage, temp_storage_bytes);
    //
    size_t bdx = 256;
    size_t gdx = (Capacity + bdx - 1) / bdx;

    collect_kv_in_ht<<<gdx, bdx, 0, stream>>>
                    (ht_keys,
                     ht_vals,
                     indicator,
                     groupby_keys,
                     agg_vals,
                     Capacity,
                     empty_key);

    #ifdef MEASURE_TIME_2
    cudaStreamSynchronize(stream);
    timer.stop();
    timer.print_elapsed_time("collect_kv_in_ht");
    #endif
    //

    #ifdef MEASURE_TIME_2
    timer.start();
    #endif
    par_result_kv_num[par_ind] = already_load_kv_num;
    cudaMemcpyAsync(host_groupby_keys_result + par_kv_begin[par_ind], groupby_keys, already_load_kv_num * sizeof(key_type), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(host_agg_vals_result + par_kv_begin[par_ind], agg_vals, already_load_kv_num * sizeof(val_type), cudaMemcpyDeviceToHost, stream);

    #ifdef MEASURE_TIME_2
    cudaStreamSynchronize(stream);
    timer.stop();
    timer.print_elapsed_time("copy result to host");
    #endif
  }
  //
}



void groupby_agg_intra_partition(std::vector<par_result> &par_result_vec,
                                      key_type *&host_groupby_keys_result,
                                      val_type *&host_agg_vals_result,
                                      size_t Capacity,
                                      size_t min_load_num,
                                      size_t max_load_num,
                                      size_t nstreams,
                                      std::vector<size_t> &par_kv_begin,
                                      std::vector<size_t> &par_result_kv_num)
{
  // Merge partitions to make the partition size as close to K as possible 
  par_result_vec = merge_partition(par_result_vec, min_load_num);
  //

  // decide number of cuda stream (one stream for one partition)
  auto par_num = par_result_vec.size();
  nstreams = nstreams < par_num ? nstreams : par_num;
  //

  // decide max result kv num
  size_t max_kv_num = 0;
  // std::vector<size_t> par_kv_begin;
  // std::vector<size_t> par_result_kv_num(par_num);
  par_kv_begin.reserve(par_num);
  par_result_kv_num.resize(par_num);
  for (size_t i=0; i<par_num; i++) {
    par_kv_begin.push_back(max_kv_num);
    max_kv_num += par_result_vec[i].size;
  }
  //

  // allocate mem for result in host
  cudaMallocHost(&host_groupby_keys_result, max_kv_num * sizeof(key_type));
  cudaMallocHost(&host_agg_vals_result, max_kv_num * sizeof(val_type));
  //

  // statistic real-time result num
  u_int32_t *stream_real_time_result_num;
  cudaMallocHost(&stream_real_time_result_num, nstreams * sizeof(u_int32_t));
  //

  // allocate device memory
  key_type **groupby_keys = (key_type **)malloc(sizeof(key_type *) * nstreams);
  val_type **agg_vals = (val_type **)malloc(sizeof(val_type *) * nstreams);
  key_type **ht_keys = (key_type **)malloc(sizeof(key_type *) * nstreams);
  val_type **ht_vals = (val_type **)malloc(sizeof(val_type *) * nstreams);
  u_int32_t **indicator = (u_int32_t **)malloc(sizeof(u_int32_t *) * nstreams);
  void **temp_storage = (void **)malloc(sizeof(void *) * nstreams);
  u_int32_t **block_result_num = (u_int32_t **)malloc(sizeof(u_int32_t *) * nstreams);

  size_t block_threads_num = 256;
  size_t max_thread_blocks_num = (Capacity + block_threads_num - 1) / block_threads_num;

  void *d_temp_storage = NULL;
  u_int32_t *d_in;
  size_t temp_storage_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_in, Capacity);
  auto residual = temp_storage_bytes % 8;
  temp_storage_bytes += (8-residual);
  

  auto dev_ptr = pre_alloc_device_memory(groupby_keys,
                                         agg_vals,
                                         ht_keys,
                                         ht_vals,
                                         indicator,
                                         temp_storage,
                                         block_result_num,
                                         Capacity,
                                         max_load_num,
                                         temp_storage_bytes,
                                         max_thread_blocks_num,
                                         nstreams);
  //

  // create nstreams threads to deal with par_num pars, each thread is bound to a cuda stream
  cudaStream_t *streams = (cudaStream_t *)malloc(nstreams * sizeof(cudaStream_t));
  for (size_t i = 0; i < nstreams; i++) 
  {
    cudaStreamCreateWithFlags(streams + i, cudaStreamNonBlocking);
  }


  RuntimeMeasurement timer_phase2;
  timer_phase2.start();


  std::vector<std::thread> nthreads(nstreams);
  g_par_counter = 0;

  for (int i = 0; i < nstreams; i++) {
    nthreads[i] = std::thread(groupby_agg_intra_partition_thread, 
                              std::ref(par_result_vec), 
                              groupby_keys[i], 
                              agg_vals[i], 
                              ht_keys[i], 
                              ht_vals[i],
                              indicator[i],
                              temp_storage[i],
                              block_result_num[i],
                              host_groupby_keys_result,
                              host_agg_vals_result,
                              Capacity,
                              temp_storage_bytes,
                              max_load_num,
                              min_load_num,
                              streams[i],
                              stream_real_time_result_num + i,
                              par_num,
                              std::ref(par_kv_begin),
                              std::ref(par_result_kv_num));               
  }

  /// assign cpu task
  // size_t cpu_groupby_agg_threads_num = 4;
  // std::vector<std::thread> cpu_groupby_agg_threads(cpu_groupby_agg_threads_num);
  // for (size_t i = 0; i < cpu_groupby_agg_threads_num; i++)
  // {
  //   cpu_groupby_agg_threads[i] = std::thread(cpu_groupby_agg_intra_partition_thread,
  //                                            std::ref(par_result_vec),
  //                                            par_num,
  //                                            host_groupby_keys_result,
  //                                            host_agg_vals_result,
  //                                            std::ref(par_kv_begin),
  //                                            std::ref(par_result_kv_num));
  // }
  ///


  for (int i = 0; i < nstreams; i++) {
    nthreads[i].join();
  }

  // for (size_t i = 0; i < cpu_groupby_agg_threads_num; i++)
  // {
  //   cpu_groupby_agg_threads[i].join();
  // }

  timer_phase2.stop();
  timer_phase2.print_elapsed_time("phase 2");

  for (int i = 0; i < nstreams; i++) {
    cudaStreamSynchronize(streams[i]);
    cudaStreamDestroy(streams[i]);
  }

  // free space
  cudaFree(dev_ptr);
  cudaFreeHost(stream_real_time_result_num);
  free(groupby_keys);
  free(agg_vals);
  free(ht_keys);
  free(ht_vals);
  free(indicator);
  free(temp_storage);
  free(block_result_num);
  free(streams);
  //
}