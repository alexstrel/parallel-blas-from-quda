#pragma once

#include <vector>
#include <enum_quda.h>
#include <complex_quda.h>

/**
   @file transform_reduce.h

   @brief QUDA reimplementation of thrust::transform_reduce as well as
   wrappers also implementing thrust::reduce.
 */

namespace quda
{
  /**
     @brief QUDA implementation providing thrust::transform_reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory.
     @param[in] execution policy (currently location where the computation will take place)
     @param[in] begin iterator
     @param[in] end iterator     
     @param[in] init Results is initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element     
     @param[in] transformer Functor that applies transform to each element
   */ 
  //template <typename policy_t, typename reducer, typename T, typename iter_t, typename transformer>
  //typename reducer::reduce_t transform_reduce(policy_t &policy, iter_t begin_it, iter_t end_it, transformer h);
  
  template <typename policy_t, typename reduce_t, typename T, typename iter_t, typename reducer, typename transformer>
  reduce_t transform_reduce(policy_t &policy, iter_t begin_it, iter_t end_it, reduce_t init, reducer r, transformer h);
  
} // namespace quda
