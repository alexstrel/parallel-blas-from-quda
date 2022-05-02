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
  template double transform_reduce<QudaFieldLocation, double, float, iter_f32_t, plus<double>, identity<float>>(
    QudaFieldLocation&, iter_f32_t, iter_f32_t, double, plus<double>, identity<float>);
} // namespace quda
