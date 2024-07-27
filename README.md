# cpu_gpu_coprocess_groupby_agg

- [] lf_max, lf_min to calculate max_load_num, min_load_num
- [] temp_storage_bytes precisely calculate
- [] multiple thread_block_num
- [] shared mem hashtable pre agg
- [] support multi data type and don't overflow
- [] for every gpu kernel, you would better check last block for boundary
- [] c++ scope, global variable
- [] local hash table max probe frequency
- [] attention data initialize before every gpu kernel
- [] attention reduce kernel one thread deal with two item
- [] hash function random seed
- [] how to set tile_len
- [] attention alignment problem