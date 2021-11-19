#pragma once

namespace quda {
  /** comm reducer for doing summation inter-process reduction */
  template <typename T> struct comm_reduce_sum {
    // FIXME - this will break when we have non-double reduction types, e.g., double-double on the host
    void operator()(std::vector<T> &v) { comm_allreduce_array(reinterpret_cast<double*>(v.data()), v.size() * sizeof(T) / sizeof(double)); }
  };
  /** comm reducer for doing max inter-process reduction */
  template <typename T> struct comm_reduce_max {
    // FIXME - this will break when we have non-double reduction types, e.g., double-double on the host
    void operator()(std::vector<T> &v) { comm_allreduce_max_array(reinterpret_cast<double*>(v.data()), v.size() * sizeof(T) / sizeof(double)); }
  };

  /** comm reducer for doing min inter-process reduction */
  template <typename T> struct comm_reduce_min {
    // FIXME - this will break when we have non-double reduction types, e.g., double-double on the host
    void operator()(std::vector<T> &v) { comm_allreduce_min_array(reinterpret_cast<double*>(v.data()), v.size() * sizeof(T) / sizeof(double)); }
  };

  /////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////

  template <> struct get_comm_reducer_t<double, plus<double>> { using type = comm_reduce_sum<double>; };
  //NEW
  template <> struct get_comm_reducer_t<float,  plus<float>>  { using type = comm_reduce_sum<float> ; };
}

