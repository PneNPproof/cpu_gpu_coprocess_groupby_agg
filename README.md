# cpu_gpu_coprocess_groupby_agg

- [ ] lf_max, lf_min to calculate max_load_num, min_load_num
- [ ] temp_storage_bytes precisely calculate
- [ ] multiple thread_block_num
- [ ] shared mem hashtable pre agg
- [ ] support multi data type and don't overflow
- [ ] for every gpu kernel, you would better check last block for boundary
- [ ] c++ scope, global variable
- [ ] local hash table max probe frequency
- [ ] attention data initialize before every gpu kernel
- [ ] attention reduce kernel one thread deal with two item
- [ ] hash function random seed
- [ ] how to set tile_len
- [ ] attention alignment problem
- [ ] cpu process partition only or groupby_agg_partiton in phase_1
- [ ] cpu collect into host_key_buffer(the first tile cpu process need to copy to a extra buffer, other tile cpu process could be collect to last tile cpu process)
- [ ] seem to have bugs for small data
- [ ] if cpu gpu murmur3hash same
- [x] data generate algorithm need to be adjusted to generate larger range data(seem to lead to result bug for 1e10 size kv and 1e9 size cardinality)
- [ ] smaller data chunk for more balanced work load for every thread
- [ ] par_result_in_continous_mem with all cpu threads before second phase
## project log
### 20240731
1. cpu_partition_thread global_par_rec_num only one for many tile, but update par_result_vec require global_par_rec_num, but don't ensure that next tile use global_par_rec_num after used by update par_result_vec  
2. data generator expand to support very large kv_num and cardinality
3. 1e10 kv 1e9 cardinality test passed for gpu process
4. test data for configuration in 3

| method   | phase 1 time (us) | phase 2 time (us) |
|--------|-------------------|-------------------|
|gpu process|10743459|6963389|
|coprocess|9337422|6947422|


