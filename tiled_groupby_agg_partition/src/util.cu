#include "util.cuh"
#include <cuda_runtime.h>



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
                       size_t nstreams)
{
  auto total_size = ((((sizeof(key_type) *3 + sizeof(val_type) * 2) + sizeof(u_int32_t)) * tile_len) 
                    + sizeof(u_int32_t) * P) * nstreams;
  void *dev_ptr;
  cudaMalloc(&dev_ptr, total_size);
  size_t offset = 0;
  for (size_t i = 0; i < nstreams; i++)
  {
    groupby_keys[i] = (key_type *)(dev_ptr + offset);
    offset += sizeof(key_type) * tile_len;
    agg_vals[i] = (val_type *)(dev_ptr + offset);
    offset += sizeof(val_type) * tile_len;
    ht_keys[i] = (key_type *)(dev_ptr + offset);
    offset += sizeof(key_type) * tile_len;
    ht_vals[i] = (val_type *)(dev_ptr + offset);
    offset += sizeof(val_type) * tile_len;
    indicator[i] = (u_int32_t *)(dev_ptr + offset);
    offset += sizeof(u_int32_t) * tile_len;
    par_rec_num[i] = (u_int32_t *)(dev_ptr + offset);
    offset += sizeof(u_int32_t) * P;
    temp_store[i] = dev_ptr + offset;
    offset += sizeof(key_type) * tile_len;
  }

  return dev_ptr;
}



template<typename key_type, typename val_type>
void scan_inclusive(key_type *indicator, 
                    val_type *indicator_scan, 
                    size_t n,
                    cudaStream_t stream,
                    void *temp_store,
                    size_t temp_store_bytes) 
{
    
    // void            *d_temp_storage = NULL;
    // size_t          temp_storage_bytes = 0;
    // CubDebugExit(cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, indicator, indicator_scan, n, stream));
    // CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes, stream));
    // printf("hello\n");
    // Run
    (cub::DeviceScan::InclusiveSum(temp_store, temp_store_bytes, indicator, indicator_scan, n, stream));

    // if (d_temp_storage)
    // {
    //   CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    // }
}


template<typename key_type, typename val_type>
void scan_exclusive(key_type *indicator, 
                    val_type *indicator_scan, 
                    size_t n,
                    cudaStream_t stream,
                    void *temp_store,
                    size_t temp_store_bytes)
{
    // void            *d_temp_storage = NULL;
    // size_t          temp_storage_bytes = 0;
    // CubDebugExit(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, indicator, indicator_scan, n, stream));
    // CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes, stream));

    // Run
    (cub::DeviceScan::ExclusiveSum(temp_store, temp_store_bytes, indicator, indicator_scan, n, stream));

    // if (d_temp_storage)
    // {
    //   CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    // }  
}

template
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

template
void scan_inclusive(key_type *indicator, 
                    val_type *indicator_scan, 
                    size_t n,
                    cudaStream_t stream,
                    void *temp_store,
                    size_t temp_store_bytes);

template
void scan_exclusive(key_type *indicator, 
                    val_type *indicator_scan, 
                    size_t n,
                    cudaStream_t stream,
                    void *temp_store,
                    size_t temp_store_bytes);