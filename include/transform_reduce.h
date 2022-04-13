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
  template <typename policy_t, typename reducer, typename T, typename iter_t, typename transformer>
  typename reducer::reduce_t transform_reduce(policy_t &policy, iter_t begin_it, iter_t end_it, transformer h);
  
  /**
     @brief QUDA implementation providing thrust::transform_reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory.
     @param[in] execution policy (currently location where the computation will take place)
     @param[in] begin iterator (first vector)
     @param[in] end iterator  (first vector)   
     @param[in] begin iterator (second vector)     
     @param[in] init Results is initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element     
     @param[in] transformer Functor that applies transform to each element
   */ 
  template <typename policy_t, typename reducer, typename T, typename iter_t, typename transformer>
  typename reducer::reduce_t transform_reduce(policy_t &policy, iter_t begin_it1, iter_t end_it1, iter_t begin_it2, transformer h);
  
    /**
     @brief QUDA implementation providing thrust::transform like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory.
     @param[in] execution policy (currently location where the computation will take place)
     @param[in] begin iterator
     @param[in] end iterator     
     @param[in] init Results is initialized to this value  
     @param[in] transformer Functor that applies transform to each element
   */ 
  template <typename policy_t, typename iter_t, typename iter2_t, typename transformer>
  iter2_t transform(policy_t &policy, iter_t begin_it, iter_t end_it, iter2_t d_first, transformer h);
  
  /**
     @brief QUDA implementation providing thrust::transform_reduce like
     functionality.  Improves upon thrust's implementation since a
     single kernel is used which writes the result directly to host
     memory.
     @param[in] execution policy (currently location where the computation will take place)
     @param[in] begin iterator (first vector)
     @param[in] end iterator  (first vector)   
     @param[in] begin iterator (second vector)     
     @param[in] init Results is initialized to this value
     @param[in] reducer Functor that applies the reduction to each transformed element     
     @param[in] transformer Functor that applies transform to each element
   */ 
  template <typename policy_t, typename reducer, typename iter_t, typename iter2_t, typename iter3_t, typename transformer>
  iter3_t transform(policy_t &policy, iter_t begin_it1, iter_t end_it1, iter2_t begin_it2, iter3_t d_first, transformer h);

  
} // namespace quda
