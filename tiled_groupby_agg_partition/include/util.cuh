#pragma once

#include <stdlib.h>
#include <cub/cub.cuh>

#include <chrono>
#include <string>

#include "MurmurHash3.h"

class RuntimeMeasurement {
public:
    inline void start() {
        start_time = std::chrono::steady_clock::now();
    }

    inline void stop() {
        end_time = std::chrono::steady_clock::now();
    }

    inline void print_elapsed_time(std::string info) {
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << info << " elapsed time: " << duration.count() << " microseconds\n";
    }

private:
    std::chrono::steady_clock::time_point start_time, end_time;
};

class murmur3_32bit {
public:
    murmur3_32bit(u_int32_t random_seed);
    u_int32_t operator()(key_type key) const;
private:
    u_int32_t seed;
};

typedef u_int32_t key_type;
typedef u_int32_t val_type;
const key_type empty_k = 0xffffffff;

template<typename key_type, typename val_type>
void *pre_device_alloc(key_type **groupby_keys,
                       val_type **agg_vals,
                       key_type **ht_keys,
                       val_type **ht_vals,
                       u_int32_t **indicator,
                       u_int32_t **par_rec_num,
                       void **temp_store,
                       size_t tile_len,
                       size_t P,
                       size_t nstreams);

template<typename key_type, typename val_type>
void scan_inclusive(key_type *indicator, 
                    val_type *indicator_scan, 
                    size_t n,
                    cudaStream_t stream,
                    void *temp_store,
                    size_t temp_store_bytes);

template<typename key_type, typename val_type>
void scan_exclusive(key_type *indicator, 
                    val_type *indicator_scan, 
                    size_t n,
                    cudaStream_t stream,
                    void *temp_store,
                    size_t temp_store_bytes);