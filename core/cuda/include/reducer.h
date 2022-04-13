#pragma once

// base reduction operations
namespace quda {

  template <typename T> struct plus {
    static constexpr bool do_sum = true;
    using reduce_t = T;
    using reducer_t = plus<T>;
    template <typename U> static inline void comm_reduce(std::vector<U> &a) { comm_allreduce_sum(a); }
    __device__ __host__ static inline T init() { return zero<T>(); }
    __device__ __host__ static inline T apply(T a, T b) { return a + b; }
    __device__ __host__ inline T operator()(T a, T b) const { return apply(a, b); }
 
  };

  template <typename T> struct maximum {
    static constexpr bool do_sum = false;
    using reduce_t = T;
    using reducer_t = maximum<T>;
    template <typename U> static inline void comm_reduce(std::vector<U> &a) { comm_allreduce_max(a); }
    __device__ __host__ static inline T init() { return low<T>::value(); }
    __device__ __host__ static inline T apply(T a, T b) { return max(a, b); }
    __device__ __host__ inline T operator()(T a, T b) const { return apply(a, b); }
  };

  template <typename T> struct minimum {
    static constexpr bool do_sum = false;
    using reduce_t = T;
    using reducer_t = minimum<T>;
    template <typename U> static inline void comm_reduce(std::vector<U> &a) { comm_allreduce_min(a); }
    __device__ __host__ static inline T init() { return high<T>::value(); }
    __device__ __host__ static inline T apply(T a, T b) { return min(a, b); }
    __device__ __host__ inline T operator()(T a, T b) const { return apply(a, b); }
  };
  
  

  template<typename T> struct cplus{
    static constexpr bool do_sum = true;
    using reduce_t = quda::complex<T>;
    using reducer_t = plus<quda::complex<T>>;
    
    template <typename U> static inline void comm_reduce(std::vector<quda::complex<U>> &a) { comm_allreduce_sum(a); }
    __device__ __host__ static inline quda::complex<T> init() { return zero<quda::complex<T>>(); }
    __device__ __host__ static inline quda::complex<T> apply(quda::complex<T> a, quda::complex<T> b) { return quda::complex<T>(a.real()+b.real(), a.imag()+b.imag()); }
    __device__ __host__ inline quda::complex<T> operator()(quda::complex<T> a, quda::complex<T> b) const { return apply(a, b); }
    
  };


  template<typename ReduceType, typename Float> struct square_ {
    square_(ReduceType = 1.0) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<Float> &x) const
    { return static_cast<ReduceType>(norm(x)); }
  };
  

  template <typename ReduceType> struct square_<ReduceType, int8_t> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<int8_t> &x) const
    { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
  };

  template<typename ReduceType> struct square_<ReduceType,short> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<short> &x) const
    { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
  };

  template<typename ReduceType> struct square_<ReduceType,int> {
    const ReduceType scale;
    square_(const ReduceType scale) : scale(scale) { }
    __host__ __device__ inline ReduceType operator()(const quda::complex<int> &x) const
    { return norm(scale * complex<ReduceType>(x.real(), x.imag())); }
  };

  template<typename Float, typename storeFloat> struct abs_ {
    abs_(const Float = 1.0) { }
    __host__ __device__ Float operator()(const quda::complex<storeFloat> &x) const
    { return abs(x); }
  };

  template <typename Float> struct abs_<Float, int8_t> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<int8_t> &x) const
    { return abs(scale * complex<Float>(x.real(), x.imag())); }
  };

  template<typename Float> struct abs_<Float,short> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<short> &x) const
    { return abs(scale * complex<Float>(x.real(), x.imag())); }
  };

  template<typename Float> struct abs_<Float,int> {
    Float scale;
    abs_(const Float scale) : scale(scale) { }
    __host__ __device__ Float operator()(const quda::complex<int> &x) const
    { return abs(scale * complex<Float>(x.real(), x.imag())); }
  };
}
