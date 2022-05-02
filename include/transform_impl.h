#pragma once
#include <iterators.h>
#include <transform_reduce.h>
#include <tunable_nd.h>
#include <kernels/transform_reduce.h>
#include <array>

namespace quda
{
  template <typename policy_t, int n_batch_, typename transformer>
  class Transform : public TunableGridStrideKernel2D
  {
    using Arg = TransformArg<n_batch_, transformer>;
    
    policy_t policy;

    int n_items;
    transformer h;        

    bool tuneSharedBytes() const { return false; }

    void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      param.grid.y = n_batch_;
    }
    // for these streaming kernels, there is no need to tune the grid size, just use max
    unsigned int minGridSize() const { return maxGridSize(); }

  public:
  
    Transform(policy_t &policy, int n_items, transformer h) :
      TunableGridStrideKernel2D(n_items, n_batch_, policy),//size_t n_items, unsigned int vector_length_y, policy keeps location
      policy(policy),
      n_items(n_items),
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
      Arg arg(n_items, h);
      //
      launch<transform_, true>(tp, stream, arg);
    }

    long long bytes() const { return n_batch_ * n_items; }//?? need to deduce from h
  };

  template <typename policy_t, typename iter_t, typename iter2_t, typename transformer>
  iter2_t transform(policy_t &policy, iter_t begin_it1, iter_t end_it1, iter2_t begin_it2, transformer h)
  {
    constexpr int n_batch = 1;
    //
    const int n_items = end_it1 - begin_it1;

    Transform<policy_t, n_batch, transformer> transform(policy, n_items, h);
    
    //if constexpr (!is_async) policy.get_queue().wait();
    
    return begin_it2+n_items;
  }  
  
  template <typename policy_t, typename reduce_t, typename iter_t, typename iter2_t, typename iter3_t, typename transformer>
  iter3_t transform(policy_t &policy,  iter_t begin_it1, iter_t end_it1, iter2_t begin_it2, iter3_t d_first, transformer h)
  {
    constexpr int n_batch = 1;
    //
    const int n_items = end_it1 - begin_it1;

    Transform<policy_t, n_batch, transformer> transform(policy, n_items, h);
    
    //if constexpr (!is_async) policy.get_queue().wait();
    
    return d_first+n_items;
  }


} // namespace quda
