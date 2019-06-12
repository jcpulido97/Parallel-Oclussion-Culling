#pragma once

#include <cuda_runtime.h>
#include <deque>
#include <chrono>
#include <math.h>
#include <iostream>
#include "octreeobb.h"
#include "../libs/MathGeoLib.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

struct vec{
  float x, y, z;
};

class CudaQuery{
  private:
    std::pair<unsigned int, vec[8]>* gpu_points;
    std::pair<unsigned int, vec[8]>* cpu_points;
    unsigned int* object_visibility_array;
    int size;
  public:
    // CudaQuery(const std::vector<OctreeOBB>& objects);
    CudaQuery(const std::deque<OctreeOBB>& objects);
    ~CudaQuery(){gpuErrchk(cudaFree(gpu_points));
                 gpuErrchk(cudaFree(object_visibility_array));
                 delete[] cpu_points;};
    void transferGPU();
    void run(vec camera_pos, vec camera_dir, std::vector<unsigned int>* cpu_visibility);
    void allocate(const std::vector<OctreeOBB>& objects);
};
