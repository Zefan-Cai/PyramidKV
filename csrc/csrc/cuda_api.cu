#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/python.h>

#include "cuda_api.h"
#include "static_switch.h"

template<typename tensor_t, int kblock_size=1024>
__global__ void update_flatten_view_kernel(tensor_t* dst_ptr, tensor_t* src_ptr, tensor_t* state_ptr, int* headlens,
        int *cu_headlens,
        int dim) {
    // Create new tensor from cache and insert element into it.

    int head_idx = blockIdx.x;
    int thread_group = blockIdx.y;
    int tid = threadIdx.x + thread_group * blockDim.x;
    int num_threads = blockDim.x * gridDim.y;

    int headlen = headlens[head_idx];

    // get position of src, dst, insert ptr
    int src_cum_off = cu_headlens[head_idx] * dim;
    int dst_cum_off = src_cum_off + head_idx * dim;
    int insert_off = (cu_headlens[head_idx+1] + head_idx) * dim;

    auto old_cache_ptr = src_ptr + src_cum_off;
    auto new_cache_ptr = dst_ptr + dst_cum_off;
    auto insert_cache_ptr = dst_ptr + insert_off;

    // copy old data
    for (int start_addr = 0; start_addr < headlen * dim; start_addr += kblock_size * num_threads) {
        auto src_addr = old_cache_ptr + start_addr + tid * kblock_size;
        auto dst_addr = new_cache_ptr + start_addr + tid * kblock_size;

        #pragma unroll
        for (int i = 0; i < kblock_size; i++) {
            if (start_addr + tid * kblock_size + i >= headlen * dim) {
                return;
            }
            dst_addr[i] = src_addr[i];
        }
    }

    // insert new data
    if (tid < dim) {
        auto insert_src_ptr = state_ptr + head_idx * dim + tid;
        auto insert_dst_addr = insert_cache_ptr + tid;
        *insert_dst_addr = *insert_src_ptr;
    }
}

torch::Tensor update_flatten_view(torch::Tensor &cache, torch::Tensor &state, torch::Tensor &headlens, torch::Tensor& cu_headlens) {
  TORCH_CHECK(headlens.dtype() == torch::kInt32, "expected headlens to be int32");
  TORCH_CHECK(cu_headlens.dtype() == torch::kInt32, "expected cu_dst_pos to be int32");

  auto cache_shape = cache.sizes();

  int origin_len = cache_shape[0];
  int head_dim = cache_shape[1];
  int head_num = headlens.sizes()[0];

  torch::Tensor out = torch::empty({origin_len + head_num, head_dim}, cache.options());

  const int kblock_size = 1;
  const int num_threads_group = 1024;
  const int num_threads = 128;

  dim3 grid(head_num, num_threads_group);

  dim3 block(num_threads);
  TORCH_CHECK(num_threads >= head_dim, "num threads should larger than head dim");

  FP16_SWITCH(cache.dtype() == torch::kFloat16, [&] {
      auto kernel = update_flatten_view_kernel<elem_type, kblock_size>;
      kernel<<<grid, block, 0>>>((elem_type*)out.data_ptr(), (elem_type*)cache.data_ptr(), (elem_type*)state.data_ptr(), (int*)headlens.data_ptr(), (int*)cu_headlens.data_ptr(), head_dim);
  });

  // TODO: when to use sync or torch auto
  // cudaDeviceSynchronize();

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("package_name", &function_name, "function_docstring"")

  m.def("update_flatten_view", &update_flatten_view, "update flatten view cache");
}
