#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <stack>
#include <vector>
#include <mutex>

#include "util.cuh"

class par_result {
 public:
  std::vector<key_type *> key;
  std::vector<val_type *> val;
  std::vector<uint32_t> end_indicator;
  uint32_t size;
  std::mutex par_lock;

  par_result() {
    size = 0;
  }
 
  par_result(const par_result &pr){
    key = pr.key;
    val = pr.val;
    end_indicator = pr.end_indicator;
    size = pr.size;
  }

  par_result & operator=(const par_result &pr)  {
    key = pr.key;
    val = pr.val;
    end_indicator = pr.end_indicator;
    size = pr.size;
    return *this;
  }

  // thread safe push
  void push(key_type *tile_par_keys, val_type *tile_par_vals, uint32_t n) {
    par_lock.lock();
    key.push_back(tile_par_keys);
    val.push_back(tile_par_vals);
    end_indicator.push_back(n);
    size+=n;
    par_lock.unlock();
  }

  void pop2device(key_type *device_keys, val_type *device_vals, uint32_t n, cudaStream_t stream) {
    while (n != 0) {
      auto slide_size = end_indicator.back();
      auto slide_keys = key.back();
      auto slide_vals = val.back();
      if (slide_size > n) {
        auto pop_keys = slide_keys + slide_size - n;
        auto pop_vals = slide_vals + slide_size - n;
        cudaMemcpyAsync(device_keys, pop_keys, n * sizeof(key_type), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_vals, pop_vals, n * sizeof(val_type), cudaMemcpyHostToDevice, stream);
        (end_indicator.back()) = slide_size - n;
        size -= n;
        n = 0;
      }
      else {
        cudaMemcpyAsync(device_keys, slide_keys, slide_size * sizeof(key_type), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(device_vals, slide_vals, slide_size * sizeof(val_type), cudaMemcpyHostToDevice, stream);
        key.pop_back();
        val.pop_back();
        end_indicator.pop_back();
        size -= slide_size;
        n -= slide_size;
        device_keys += slide_size;
        device_vals += slide_size;
      }
    }
  }

  void merge(const par_result &pr){
    key.insert(key.end(), pr.key.begin(), pr.key.end());
    val.insert(val.end(), pr.val.begin(), pr.val.end());
    end_indicator.insert(end_indicator.end(), pr.end_indicator.begin(), pr.end_indicator.end());
    size += pr.size;
  }

  uint32_t Size() { return size; }
};

void update_par_result_task(key_type *tile_key_buffer,
                            val_type *tile_val_buffer, 
                            uint32_t *tile_par_pos, 
                            size_t P,
                            std::vector<par_result> &par_result_vec);