#pragma once

#include "ToucanGPUGenDataTypes.h"
#include "SimConfigs.h"

// #include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s, file: %s, line: %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void evalSingleCycle();

__global__ void evalFreeRunningNCycles(uint32_t cycleCnt);
__global__ void evalFreeRunningNCycles_Large(uint32_t cycleCnt);

uint64_t read_reg_from_gpu(const std::vector<std::tuple<uint32_t, uint32_t>>& signalLocs);

void write_reg_to_gpu(const std::vector<std::tuple<uint32_t, uint32_t>>& signalLocs, uint64_t signalValue);

void copy_netlist_to_gpu(toucanGPUSim::SimDesignInfo &design);

bool get_eval_done();

uint32_t get_real_cycles();

void setEnablePrint(bool print_en);

#ifdef ENABLE_SIM_PROFILE
void copy_profile_data(int64_t *usefulTicks, int64_t *totalTicks, size_t count);
#endif
