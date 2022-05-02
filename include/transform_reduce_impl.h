#pragma once
#include <comm_quda.h>
#include <float_vector.h>
#include <array>
#include <iterators.h>
#include <reduce_helper.h>
#include <transform_reduce.h>
#include <tunable_reduction.h>
#include <kernels/transform_reduce.h>

namespace quda
{

  /**
     Trait that returns the correct comm reduce class for a given reducer :: custom global reducers moved to global_reducer.h
   */
  template <typename T, typename reducer> struct get_comm_reducer_t { };

  template <typename policy_t, typename T, int n_batch_, typename reducer, typename transformer>
  class TransformReduce : TunableMultiReduction<1>
  {
    using reduce_t = typename reducer::reduce_t;
    using Arg = TransformReduceArg<T, n_batch_, reducer, transformer>;
    
    policy_t policy;
    std::vector<reduce_t> &result;
    int n_items;
    reduce_t init;
    reducer r;
    transformer h;        

    bool tuneSharedBytes() const { return false; }

  public:
  
    TransformReduce(policy_t &policy, std::vector<reduce_t> &result, int n_items, reduce_t init, reducer r, transformer h) :
      TunableMultiReduction(n_items, n_batch_, Arg::max_n_batch_block, policy),//policy keeps location
      policy(policy),
      result(result),
      n_items(n_items),
      init(init),
      r(r),
      h(h)      
    {
      strcpy(aux, "batch_size=");
      u32toa(aux + 11, n_batch_);
      apply(device::get_default_stream());
    }

    void apply(const qudaStream_t &stream)
    {
      TuneParam tp = tuneLaunch(*this, getTuning(), getVerbosity());
      //
      Arg arg(n_items, init, h);
      //
      launch<transform_reducer, true>(result, tp, stream, arg);
    }

    long long bytes() const { return n_batch_ * n_items * sizeof(T); }//need to deduce from h
  };

  template <typename policy_t, typename reduce_t, typename T, typename iter_t, typename reducer, typename transformer>
  reduce_t transform_reduce(policy_t &policy, iter_t begin_it, iter_t end_it, reduce_t init, reducer r, transformer h)
  {
    constexpr int n_batch = 1;
    std::vector<reduce_t> result = {0.0};
    const int n_items = end_it - begin_it;

    TransformReduce<policy_t, T, n_batch, reducer, transformer> transformReducer(policy, result, n_items, init, r, h);
    
    //if constexpr (!is_async) policy.get_queue().wait();
    
    return result[0];
  }  

} // namespace quda
