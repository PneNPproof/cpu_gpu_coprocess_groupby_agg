#include <par_result.cuh>

void update_par_result_task(key_type *tile_key_buffer,
                            val_type *tile_val_buffer, 
                            uint32_t *tile_par_pos, 
                            size_t P,
                            std::vector<par_result> &par_result_vec)
{
  for (size_t par_ind = 0; par_ind < P; par_ind++) {
    par_result_vec[par_ind].push(tile_key_buffer + tile_par_pos[par_ind], tile_val_buffer + tile_par_pos[par_ind],
                                 tile_par_pos[par_ind + 1] - tile_par_pos[par_ind]);
  }  
}