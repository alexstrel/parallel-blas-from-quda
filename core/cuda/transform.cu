#include <transform_impl.h>
//target specific implementations:
#include <transformer.h>

namespace quda
{
  using iter_f32_t = decltype(std::vector<float, AlignedAllocator<float>>().begin());
  using iter_c32_t = decltype(std::vector<quda::complex<float>, AlignedAllocator<quda::complex<float> >>().begin()); 

  // explicit instantiation list for transform_reduce
  template iter_f32_t transform<QudaFieldLocation, iter_f32_t, iter_f32_t, axpyDot<float>>(
      QudaFieldLocation&, iter_f32_t, iter_f32_t, iter_f32_t, axpyDot<float>); 
      
} // namespace quda