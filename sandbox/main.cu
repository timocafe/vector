// vector library header file.

// Copyright (c) 2021 Timothée Ewart

// Use, modification, and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#include <algorithm>
#include <chrono>
#include <iostream>

#include "memory/vector.h"

///
/// \brief sum from the stream benchmark using [] operator and function size()
/// of the class vector, the specific copy constructor must be called.
/// STREAM sum benchmark a(:) = b(:) + c(:)
///
template <class T>
__global__ void kernel_sum(vector<T> a, const vector<T> b, const vector<T> c) {
  const uint32_t size = a.size();
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (uint32_t i = tid; i < size; i += blockDim.x * gridDim.x)
    a[i] = b[i] + c[i];
}

int main(int argc, char *argv[]) {
  typedef float value_type;
  const uint32_t size = std::atoi(argv[1]);
  std::cout << " Stream benchmark SUM: a(:) = b(:) + c(:) \n";
  vector<value_type> a(size);
  vector<value_type> b(size, 1);
  vector<value_type> c(size, 2);
  // not needed but avoid memory miss
  a.prefetch_gpu();
  b.prefetch_gpu();
  c.prefetch_gpu();

  auto start = std::chrono::system_clock::now();
  kernel_sum<value_type><<<32, 256>>>(
      {policy_shared(), a}, {policy_shared(), b}, {policy_shared(), c});
  cudaDeviceSynchronize();
  auto end = std::chrono::system_clock::now();
  auto elapsed =
      std::chrono::duration<float, std::chrono::seconds::period>(end - start);

  a.prefetch_cpu();
  //
  if (std::all_of(a.begin(), a.end(), [&](value_type i) { return i == 3; })) {
    std::cout
        << " All elements are the same ! The Stream benchmark SUM works. \n";
    // 4 = 3 load and 1 write, long story
    // https://blogs.fau.de/hager/archives/8263
    std::cout << " Total memory allocated: "
              << (float)(a.memory_allocated() + b.memory_allocated() +
                         c.memory_allocated()) /
                     (1024. * 1024. * 1024.)
              << "[GB]\n";
    std::cout << " Bandwidth SUM benchmark: "
              << (4. * (a.memory_allocated()) / elapsed.count()) /
                     (1024 * 1024 * 1024)
              << "[GB/s]\n";
  } else {
    std::cout << " Oups, All elements are not the same ! \n";
  }
}
