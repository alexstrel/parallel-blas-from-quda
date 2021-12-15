#include <transform_reduce_impl.h>
//target specific implementations:
#include <reducer.h>
#include <transformer.h>
//generic implementations
#include "../generic/include/global_reducer.h"

namespace quda
{
  using iter_f32_t = decltype(std::vector<float, AlignedAllocator<float>>().begin());
  using iter_c32_t = decltype(std::vector<quda::complex<float>, AlignedAllocator<quda::complex<float> >>().begin()); 

  // explicit instantiation list for transform_reduce
  template float transform_reduce<QudaFieldLocation, float, int, plus<float>, identity<float>>(
    QudaFieldLocation&, int, int, float, plus<float>, identity<float>);

  template float transform_reduce<QudaFieldLocation, float, iter_f32_t, plus<float>, identity<float>>(
      QudaFieldLocation&, iter_f32_t, iter_f32_t, float, plus<float>, identity<float>);   
      
  template float transform_reduce<QudaFieldLocation, float, iter_f32_t, plus<float>, axpyDot<float>>(
      QudaFieldLocation&, iter_f32_t, iter_f32_t, float, plus<float>, axpyDot<float>);   

  template quda::complex<float> transform_reduce<QudaFieldLocation, quda::complex<float>, iter_c32_t, cplus<float>, caxpyDot<float>>(
      QudaFieldLocation&, iter_c32_t, iter_c32_t, quda::complex<float>, cplus<float>, caxpyDot<float>);
} // namespace quda
