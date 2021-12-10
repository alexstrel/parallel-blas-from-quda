#pragma once

//custom transform operations (DSL programming)

namespace quda {
  template <typename T> struct identity {
    static constexpr bool do_sum = false;
    T *x;
    identity(T *x_) : x(x_) {}
    T operator()(int i, int j = 0) const { return x[i]; }
  };
 
  template <typename T> struct axpyDot {
    static constexpr bool do_sum = false;  
    const T *x;
    T *y;
    const T a;

    axpyDot(const T a_, const T *x_, T *y_) : a(a_), x(x_), y(y_) {}

    T operator() (int i, int j = 0) const {
      y[i] = a*x[i] + y[i];
      return (y[i]*x[i]);
    }
  };

  template <typename T> struct caxpyDot {
    static constexpr bool do_sum = false;
    const quda::complex<T> *x;
    quda::complex<T> *y;
    const quda::complex<T> a;

    caxpyDot(const quda::complex<T> a_, const quda::complex<T> *x_, quda::complex<T> *y_) : a(a_), x(x_), y(y_) {}

    quda::complex<T> operator() (int i, int j = 0) const {
      T yr = a.real()*x[i].real() + y[i].real();
      T yi = a.imag()*x[i].imag() + y[i].imag();
      y[i] = quda::complex<T>(yr, yi);
      return quda::complex<T>(yr*x[i].real()-yi*x[i].imag(), yr*x[i].imag()-yi*x[i].real());
    }
  };


}
