#pragma once

#include <reduction_kernel.h>
#include <limits>

namespace quda {


  template <typename reducer_, typename T, int n_batch_, typename transformer_>
  struct TransformReduceArg : public ReduceArg<typename reducer_::reduce_t> {
    using reducer = reducer_;
    using reduce_t = typename reducer::reduce_t;
    using transformer = transformer_;
    static constexpr int n_batch_max = 8;
    
    int n_items;
    int n_batch;
    //T init_value;
    transformer h;

    TransformReduceArg(int n_items, transformer h) :
      ReduceArg<reduce_t>(dim3(n_items, 1, n_batch_), n_batch_),
      n_items(n_items),
      n_batch(n_batch_),
      h(h)      
    {
      if (n_batch > n_batch_max) errorQuda("Requested batch %d greater than max supported %d", n_batch, n_batch_max);
      if (n_items > std::numeric_limits<int>::max())
        errorQuda("Requested size %lu greater than max supported %lu",
                  (uint64_t)n_items, (uint64_t)std::numeric_limits<int>::max());
    }
  };

  template <typename Arg> struct transform_reducer : Arg::reducer {
    using reduce_t = typename Arg::reduce_t;
    using Arg::reducer::operator();
    using count_t = decltype(Arg::n_items);
 
    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr transform_reducer(const Arg &arg) : arg(arg) {}

    __device__ __host__ inline reduce_t operator()(reduce_t &value, count_t i, int, int j)//j is a batch indx
    {
      auto t = arg.h(i, j);
      return operator()(t, value);
    }
  };
  
template <int n_batch_, typename transformer_, bool use_kernel_arg = true>
  struct TransformArg : kernel_param<use_kernel_arg> {
    using transformer = transformer_;
    static constexpr int n_batch_max = 8;
    int n_items;
    int n_batch;
    transformer h;

    TransformArg(int n_items, transformer h) :
      kernel_param<use_kernel_arg>(dim3(n_items, n_batch_, 1)),
      n_items(n_items),
      n_batch(n_batch_),
      h(h)      
    {
      if (n_batch > n_batch_max) errorQuda("Requested batch %d greater than max supported %d", n_batch, n_batch_max);
      if (n_items > std::numeric_limits<int>::max())
        errorQuda("Requested size %lu greater than max supported %lu",
                  (uint64_t)n_items, (uint64_t)std::numeric_limits<int>::max());
    }
  };

  template <typename Arg> struct transform_ {
    using count_t = decltype(Arg::n_items);

    const Arg &arg;
    static constexpr const char *filename() { return KERNEL_FILE; }
    constexpr transform_(const Arg &arg) : arg(arg) {}

    __device__ __host__ inline void operator()(count_t i, int j){//j is a batch indx
      arg.h(i, j);
      //
      return;
    }
  };

}
