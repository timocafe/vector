
Make Oriented Object programming for Kernel (input arguments of CUDA kernel) is a nice feature, 
because C style can be painfull if the number of argument is too large. This feature can be easily done. 

In this short text, I have coded a simple [vector class](https://github.com/timocafe/vector/blob/main/memory/vector.h) which can work on CPU/GPU and facilitates the coding of CUDA kernel, and heterogeneous computing.

## What does it do ?

This tiny vector class allows computation on CPU and GPU using the same "object". It simplifies a lot the programming.
Not a big deal but simple and efficient to use. You can adapt it for any class (container) of your choice.

## What do you need ?

GCC - CUDA SDK - GPU - cmake - clangformat supporting unify memory (starting at Kelpler), preferably higher: Pascal and more.

## How does it work ?

The key stone is done into the constructor and the copy constructor and destructor.  A basic vector class needs a pointer to allocate data,
an insigned integer for the size and here we need a bool to say if the datas are on GPU (you will understand later)

The first think is to allocate memory using CUDA unify. First the basic constructor of the vector:

```cpp
  explicit vector(size_type n = 0, const value_type &v = value_type())
      : data_(nullptr), size_(n), shared_(false) { // shared is false
    cudaError_t error = cudaMallocManaged(&data_, n * sizeof(value_type));
    if (error != cudaSuccess)
      std::cout << " Error memory allocation ! \n";
    // cudaDeviceSynchronize(); needed for Jetson and K80 - Kepler
    std::fill(begin(), end(), v);
  }
```

Second a copy constructor which will be called only when we start a CUDA kernel. The object coming from this constructor
will be the object in your CUDA kernel,

```cpp
  // policy for the specific copy constructor
  struct policy_shared {};

  vector(const policy_shared &p, const vector &other) {
    data_ = other.data_;
    size_ = other.size_;
    shared_ = true; // Important the boolean shared_ becomes true
  }
```

then the destructor, the big deal. The destructor should not release the memory on the GPU (that why we need a bool).
Moreover, we check is the data_ are not null (side effect of the move constructor). 

```cpp
  ~vector() {
    if (!shared_) {
      if (data_ != nullptr) {
        cudaFree(data_);
        size_ = 0;
      }
    }
  }
```

Then you pin correctly the member function of your class with usual CUDA keyword **\_\_host\_\_ \_\_device\_\_**, per example for the bracket operator (with an assert!)

```cpp
  __host__ __device__
  inline reference operator[](size_type i) {
    assert(i < size_ && " Too ambitious! \n"); // support of the assert always usefull 
    return data_[i];
  }
```

At this point, the vector class can be used on CPU or GPU and more important, it can instantiate the kernel directly
using the copy constructor with the help of the policy:

```cpp

  // basic cuda sum benchmark between two vectors
  template <class T>
  // argument pass by copy important
  __global__ void kernel_sum(vector<T> a, const vector<T> b, const vector<T> c) {  
  const uint32_t size = a.size(); // <--- magic you get the size directly
  uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (uint32_t i = tid; i < size; i += blockDim.x * gridDim.x)
    a[i] = b[i] + c[i]; // bracket operator with the assert it can help
  }
 
  // declare your vectors
  int size = 128;
  vector<float> a(size);
  vector<float> b(size, 1);
  vector<float> c(size, 2);
  // call the CUDA kernel with the correct copy constructor
  kernel_sum<float><<<32, 256>>>({policy_shared(), a}, {policy_shared(), b}, {policy_shared(), c});
```

CUDA fanatics will complain about the page fault due to the unify memory. Do not worry data can be prefetched:

```cpp
  void prefetch_gpu(cudaStream_t s = 0) const {
    cudaMemPrefetchAsync(data(), memory_allocated(), 0,
                         s); // 0 is the default device
  }
```

## Example 

The [sandbox](https://github.com/timocafe/vector/blob/main/sandbox/main.cu) contains a STREAM benchmark, my machine is very tiny!

```bash
mkdir b
cd b
cmake -DCMAKE_BUILD_TYPE=Release ..
make

./sandbox/exe 400000000

Stream benchmark SUM: a(:) = b(:) + c(:) 
All elements are the same ! The Stream benchmark SUM works. 
Total memory allocated: 0.470348[GB]
Bandwidth SUM benchmark: 41.613[GB/s]
```

## Conclusions

With a few lines (<10!) the development of data structure and kernels can be really simplified.

## Questions
### But why not thrust ?

Thrust does many more think but conceptually it aims to parallelize common std::algorithms like sort. It also features only two containers: thrust::host_vector and thrust::device_vector. Container are specifics for device. Not in my example, and I am not competing with NVIDIA :-)

### But why not CUDA - C++17 ?

With the last realease of the CUDA SDK, [kernels can be writen directly into C++ STL::algo](https://developer.nvidia.com/blog/accelerating-standard-c-with-gpus-using-stdpar/) using a specific policy. Definitively clever and modern but I do not have the opportunity to test this SDK, and I do not know if the std::vector can be used directly into a CUDA kernel. I am not sure, but I think my approach will be compatible with c++17. Per example, in the constructor for **std::fill**, I could write:

```cpp
   std::fill(std::execution::par_unseq, begin(), end(), v); // GPU execution
```

### You have implemented a shared_ptr !

Kinda, but if I build my vector implementation on a std::shared_ptr, it will not work because the decrement of the shared ptr will be not forward from GPU to CPU during the destruction (the data members of the shared pointer have not be allocated with the unify memory)

### Nice, did you do the full STL ?

Not me but Scott Zuyderduyn did [ecuda](https://github.com/BaderLab/ecuda), it provides a lot of containers with high STD compatibility. 

### Why not provide an allocator to the std::vector ?

Because all functions of the vector will not work on the GPU, think about the keyword **\_\_host\_\_**, it does not exist into the std library. 
Moreover, **std::vector** has many mechanism of resize (think about **push_back**). Dynamic memory allocation on a GPU is impossible.
