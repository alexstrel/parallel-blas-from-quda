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
  template float transform_reduce<QudaFieldLocation, plus<float>, float, int, identity<float>>(
    QudaFieldLocation&, int, int, identity<float>);

  template float transform_reduce<QudaFieldLocation, plus<float>, float, iter_f32_t, identity<float>>(
      QudaFieldLocation&, iter_f32_t, iter_f32_t, identity<float>);   
      
  template float transform_reduce<QudaFieldLocation, plus<float>, float, iter_f32_t, axpyDot<float>>(
      QudaFieldLocation&, iter_f32_t, iter_f32_t, axpyDot<float>);   
  template quda::complex<float> transform_reduce<QudaFieldLocation, cplus<float>, quda::complex<float>, iter_c32_t, caxpyDot<float>>(
      QudaFieldLocation&, iter_c32_t, iter_c32_t, caxpyDot<float>);

  template quda::complex<float> transform_reduce<QudaFieldLocation, cplus<float>, quda::complex<float>, iter_c32_t, cDot<float>>(
      QudaFieldLocation&, iter_c32_t, iter_c32_t, cDot<float>);
} // namespace quda
