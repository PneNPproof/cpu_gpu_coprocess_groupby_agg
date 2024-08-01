#pragma once

#include <unordered_set>
#include <thread>
#include <algorithm>
#include "selfsimilar_int_distribution.h"
#include "zipfian_int_distribution.h"


template <typename Tk, typename Tv>
void generate_various_dist_kv_array(Tk *key, Tv *val, size_t cardinality, size_t kv_num, double skew_factor, std::default_random_engine &generator, int dist_kind, Tk empty_key)
{

  // printf("begin generate %ld kv of cardinality %ld of self similar dist of skew factor %f\n\n", kv_num, cardinality, skew_factor);

  // first let's generate cardinality distinct random number
  
  const auto a = (size_t)1 << (size_t)31;

  std::unordered_set<unsigned> set;

  for (size_t i = 0; i < cardinality; i++)
  {
    unsigned rn;
    unsigned before_sz;
    unsigned after_sz;

    do
    {
      before_sz = set.size();
      do
      {
        rn = (std::rand() % 2) * a + std::rand();
      } while (rn == empty_key);
      set.insert(rn);
      after_sz = set.size();
    } while (before_sz == after_sz);
  }
  ///

  // write set to key to ensure key's cardinality
  size_t counter = 0;
  for (auto u : set)
  {
    key[counter] = u;
    counter++;
  }
  ///

  // generate rest key
  
  
  std::uniform_int_distribution<unsigned> distribution_0(0, cardinality - 1);
  zipfian_int_distribution<unsigned> distribution_1(0, cardinality-1, skew_factor);
  selfsimilar_int_distribution<unsigned> distribution_2(0, cardinality - 1, skew_factor);

  for (auto i = cardinality; i < kv_num; i++)
  {
    size_t ind;
    switch (dist_kind)
    {
    case 0:
      ind = distribution_0(generator);
      break;
    case 1:
      ind = distribution_1(generator);
      break;
    default:
      ind = distribution_2(generator);
      break;
    }
    key[counter] = key[ind];
    counter++;
  }
  ///

  // generate value
  for (size_t i = 0; i < kv_num; i++)
  {
    val[i] = std::rand();
  }
  ///

  // shuffle the key
  for (size_t i=0; i<kv_num; i++) {
    auto rand_ind = (std::rand() * (size_t)(std::rand()))%kv_num;
    auto temp = key[i];
    key[i] = key[rand_ind];
    key[rand_ind] = temp;
  }

  printf("generate complete\n\n");

  // set.clear();

  // for (size_t i = 0; i < kv_num; i++)
  // {
  //   set.insert(key[i]);
  // }
  // if (set.size() == cardinality)
  // {
  //   printf("The correctness test of the generated kv passed\n\n");
  // }
  // else
  // {
  //   printf("The correctness test of the generated kv failed(in fact cardinality %ld)\n\n", set.size());
  // }
}


template <typename key_type>
void generate_m_num_in_interval(key_type interval_a, 
                                key_type interval_b, 
                                size_t m,
                                key_type *result,
                                key_type kt_max)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<key_type> distrib(0, kt_max);
  
  auto interval_len = interval_b - interval_a + 1;
  key_type last_generate_num;
  for (size_t i = 0; i < m; i++)
  {
    key_type left_end;
    key_type right_end;
    if (i == 0)
    {
      left_end = interval_a;
    }
    else
    {
      left_end = last_generate_num + 1;
    }
    right_end = interval_b - m + i + 1;
    auto len = right_end - left_end + 1;
    last_generate_num = (distrib(gen) % len) + left_end;
    result[i] = last_generate_num;
  }
}

template <typename key_type>
void generate_keys(size_t begin_ind, 
                   size_t end_ind, 
                   size_t cardinality,
                   key_type *keys)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<key_type> distrib(0, cardinality - 1);
  
  for (size_t i = begin_ind; i < end_ind; i++)
  {
    keys[i] = keys[distrib(gen)];
  }
}


template <typename val_type>
void generate_vals(size_t begin_ind, 
                   size_t end_ind, 
                   val_type vt_max,
                   val_type *vals)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<val_type> distrib(0, vt_max);
  
  for (size_t i = begin_ind; i < end_ind; i++)
  {
    vals[i] = distrib(gen);
  }
}


template <typename key_type, typename val_type>
void generate_various_dist_kv_set_multithread(key_type *keys,
                                              val_type *vals,
                                              size_t cardinality,
                                              size_t kv_num,
                                              double skew_factor,
                                              int dist_kind,
                                              key_type kt_max,
                                              val_type vt_max)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<key_type> distrib(0, kt_max);

  size_t thread_num = 8;
  std::vector<std::thread> threads;
  std::vector<size_t> thread_interval_len(thread_num);
  std::vector<size_t> thread_interval_len_scan(thread_num);
  size_t all_interval_len = kt_max + 1;

  for (size_t i = 0; i < thread_num; i++)
  {
    auto thread_interval_a = i * (all_interval_len) / thread_num;
    auto thread_interval_b = (i + 1) * (all_interval_len) / thread_num - 1;
    thread_interval_len.push_back(thread_interval_b - thread_interval_a + 1);
    all_interval_len -= thread_interval_len[i];
    thread_interval_len_scan.push_back(all_interval_len);
  }

  
  size_t already_produce = 0;
  for (size_t i = 0; i < thread_num; i++)
  {
    /// calculate how many keys each thread should produce
    size_t min_produce;
    size_t max_produce;
    if (already_produce + thread_interval_len[i] <= cardinality)
    {
      max_produce = thread_interval_len[i];
    }
    else
    {
      max_produce = cardinality - already_produce;
    }
    if (cardinality <= already_produce + thread_interval_len_scan[i])
    {
      min_produce = 0;
    }
    else
    {
      min_produce = cardinality - (already_produce + thread_interval_len_scan[i]);
    }

    auto thread_produce = min_produce + distrib(gen) % (max_produce - min_produce + 1);

    ///

    already_produce += thread_produce;
    
    auto thread_interval_a = i * (all_interval_len) / thread_num;
    auto thread_interval_b = (i + 1) * (all_interval_len) / thread_num - 1;

    threads.emplace_back(generate_m_num_in_interval<key_type>,
                         thread_interval_a,
                         thread_interval_b,
                         thread_produce,
                         keys,
                         kt_max);

  }

  for (size_t i = 0; i < thread_num; i++)
  {
    threads[i].join();
  }

  threads.clear();

  /// now generate rest keys

  size_t rest_key_num = kv_num - cardinality;

  for (size_t i = 0; i < thread_num; i++)
  {
    auto thread_keys_begin = cardinality + i * rest_key_num / thread_num;
    auto thread_keys_end = cardinality + (i + 1) * rest_key_num / thread_num;
    threads.emplace_back(generate_keys<key_type>,
                      thread_keys_begin,
                      thread_keys_end,
                      cardinality,
                      keys);
  }

  for (size_t i = 0; i < thread_num; i++)
  {
    threads[i].join();
  }

  threads.clear();

  ///

  /// now generate vals
  for (size_t i = 0; i < thread_num; i++)
  {
    auto thread_vals_begin = i * kv_num / thread_num;
    auto thread_vals_end = (i + 1) * kv_num / thread_num;
    threads.emplace_back(generate_vals<val_type>,
                      thread_vals_begin,
                      thread_vals_end,
                      vt_max,
                      vals);
  }

  for (size_t i = 0; i < thread_num; i++)
  {
    threads[i].join();
  }

  threads.clear();

  ///

  printf("generate complete\n\n");
}
