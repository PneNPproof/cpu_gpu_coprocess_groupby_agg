#pragma once

#include <unordered_set>
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
  unsigned counter = 0;
  for (auto u : set)
  {
    key[counter] = u;
    counter++;
  }
  ///

  // generate rest key
  
  
  std::uniform_int_distribution<int> distribution_0(0, cardinality - 1);
  zipfian_int_distribution<int> distribution_1(0, cardinality-1, skew_factor);
  selfsimilar_int_distribution<int> distribution_2(0, cardinality - 1, skew_factor);

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
    auto rand_ind = std::rand()%kv_num;
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