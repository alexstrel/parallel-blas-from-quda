#pragma once

#include <quda_define.h>
#include <quda_api.h>
#include <quda_constants.h>

#if defined(QUDA_TARGET_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <string>
#include <complex>
#include <vector>

// this is a helper macro for stripping the path information from
// __FILE__.  FIXME - convert this into a consexpr routine
#define KERNEL_FILE                                                                                                    \
  (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 :                                                               \
                            strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)

#define TEX_ALIGN_REQ (512*2) //Fermi, factor 2 comes from even/odd
#define ALIGNMENT_ADJUST(n) ( (n+TEX_ALIGN_REQ-1)/TEX_ALIGN_REQ*TEX_ALIGN_REQ)
//#include <quda.h>
#include <util_quda.h>
#include <malloc_quda.h>
#include <object.h>
#include <device.h>

namespace quda {

  using Complex = std::complex<double>;

  /**
   * Check that the resident gauge field is compatible with the requested inv_param
   * @param inv_param   Contains all metadata regarding host and device storage
   */
  class TimeProfile;

//from new devel (reducer.h)
  namespace reducer
  {
    /**
       @return the reduce buffer size allocated
    */
    size_t buffer_size();

    /**
       @return pointer to device reduction buffer
    */
    void *get_device_buffer();

    /**
       @return pointer to device-mapped host reduction buffer
    */
    void *get_mapped_buffer();

    /**
       @return pointer to host reduction buffer
    */
    void *get_host_buffer();
    /**
       @brief get_count returns the pointer to the counter array used
       for tracking the number of completed thread blocks.  We
       template this function, since the return type is target
       dependent.
       @return pointer to the reduction count array.
     */
    template <typename count_t> count_t *get_count();

    /**
       @return reference to the event used for synchronizing
       reductions with the host
     */
    qudaEvent_t &get_event();

    //these functions were imported from blas_quda.cu 
    void init();
    void destroy();

  } // namespace reducer

  constexpr int max_n_reduce() { return QUDA_MAX_MULTI_REDUCE; }

}

