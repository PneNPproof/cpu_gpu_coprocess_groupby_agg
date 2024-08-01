#include "util.cuh"
#include "par_result.cuh"
#include "cpu_groupby_agg_intra_partition.cuh"

#include <mutex>
#include <thread>
#include <unordered_map>

extern std::mutex g_par_counter_mutex;
extern size_t g_par_counter;

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

void cpu_groupby_agg_intra_partition_thread(std::vector<par_result> &par_result_vec,
                                            size_t par_num,
                                            key_type *host_groupby_keys_result,
                                            val_type *host_agg_vals_result,
                                            std::vector<size_t> &par_kv_begin,
                                            std::vector<size_t> &par_result_kv_num)
{
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

    auto &pr = par_result_vec[par_ind];

    auto keys = host_groupby_keys_result + par_kv_begin[par_ind];
    auto vals = host_agg_vals_result + par_kv_begin[par_ind];

    par_result_in_continous_mem(pr, keys, vals);

    std::unordered_map<key_type, val_type> ht;
    auto par_kv_num = pr.size;
    /// insert kv into ht
    for (size_t i = 0; i < par_kv_num; i++)
    {
      auto key = keys[i];
      auto val = vals[i];
      auto search = ht.find(key);
      if (search == ht.end())
      {
        ht.insert({key, val});
      }
      else
      {
        (search->second) += val;
      }
    }
    ///

    size_t par_result_num = 0;
    /// collect into result kv
    for (auto& kv : ht)
    {
      keys[par_result_num] = kv.first;
      vals[par_result_num] = kv.second;
      par_result_num++;
    }
    ///

    par_result_kv_num[par_ind] = par_result_num;
    
  }
}