// #define NDEBUG

#include "ToucanGPUGenDataTypes.h"

#include "SimEval.h"

#include <cstdint>
#include <iostream>
#include <cassert>

using namespace toucanGPUSim;
namespace cg = cooperative_groups;



// Pos for op nop: 0
// Pos for op and: 34
// Pos for op or: 290
// Pos for op xor: 546
// Pos for op rep1b: 16
// Pos for op cmp_eq: 802
// Pos for op mul_hi: 1058
// Pos for op mul_lo: 1314
// Pos for op add: 2082
// Pos for op carry: 1570
// Pos for op mux: 2594
// Pos for op dshl: 3106
// Pos for op dshr: 4130
// Pos for op xorr: 18
// Pos for op shl1: 3362
// Pos for op shl2: 3618
// Pos for op shl3: 3874
// Pos for op shr1: 0
// Pos for op shr2: 0
// Pos for op shr3: 0




// bool eval(toucanGPUSim::SimDesignInfo &design, bool enablePrint) {
//   // TODO: eval part by part
//   for (const auto &eachRegionParts: design.regionPartitionIds) {
//     for (const auto partId: eachRegionParts) {
//       auto &part = design.parts[partId];
//       evalPartL0(design, part);
//       evalExecLevels(design, part, design.lut);
//       evalLastLevel(design, part, enablePrint);
//     }
//   }
//   return design.shouldStop;
// }





#define LUT_SIZE 5154

typedef struct {
  // Private data
  uint8_t *valuePool;
  size_t numOpsL0;
  CGTopLevelMetaInfo *ops_l0;
  size_t *numOpsExec;
  CGExecLevelMetaInfo **ops_exec;
  size_t numOpsLast;
  CGLastLevelMetaInfo *ops_last;

  uint32_t numExecLevels;
} SimPartitionPtrs;


__constant__ uint8_t lutContent[LUT_SIZE];

__device__ uint8_t *regPool, *memPool, *exchangePool;

__device__ bool shouldStop = false;
__device__ bool enablePrint = false;
__device__ uint32_t realCycles;

__device__ char **printMsgs;
__device__ SimPartitionPtrs *partitions;
__device__ uint32_t **partsInRegion;
__device__ uint32_t *numPartsInRegion;
__device__ uint32_t numRegions;

uint8_t *regPool_device, *memPool_device, *exchangePool_device;

std::vector<SimPartitionPtrs> gpuPartInfos;

// every thread group
__device__ void evalPartL0(
  uint8_t * __restrict valuePool, 
  const CGTopLevelMetaInfo * __restrict topLevelOps, 
  const size_t numOps) {

  auto block = cg::this_thread_block();
  auto thread_rank = block.thread_rank();
  auto threads_in_block = block.size();


  for (size_t exec_pos = 0; exec_pos < numOps; exec_pos += threads_in_block) {
    size_t op_pos = exec_pos + thread_rank;
    if (op_pos < numOps) {
      // valid
      auto op = topLevelOps[op_pos];
      switch (op.opType) {
        case static_cast<uint8_t>(CGToucanOPName::RegRead): {
          // Reg Read
          auto regValId = op.reg.reg;
          auto resultId = op.reg.result;
          auto regVal = regPool[regValId];
          valuePool[resultId] = regVal;
          break;
        };
        case static_cast<uint8_t>(CGToucanOPName::ExchangeRead): {
          // ExchangeRead
          auto exgValId = op.exgRead.exchangeVal;
          auto localValId = op.exgRead.localVal;
          auto exgVal = exchangePool[exgValId];
          valuePool[localValId] = exgVal;
          break;
        }
        default: {
          printf("Unexpected op type: %d!\n", static_cast<uint32_t>(op.opType));
          shouldStop = true;
          assert(false && "Should not reach here");
        }
      }
    }
  }
}

__device__ void evalExecLevels(
  uint8_t * __restrict valuePool, 
  // uint8_t * __restrict exchangePool, 
  const CGExecLevelMetaInfo * __restrict execLevelOps, 
  const size_t numOps) {

  auto block = cg::this_thread_block();
  auto thread_rank = block.thread_rank();
  auto threads_in_block = block.size();


  for (size_t exec_pos = 0; exec_pos < numOps; exec_pos += threads_in_block) {
    size_t op_pos = exec_pos + thread_rank;
    if (op_pos < numOps) {
      // valid
      auto op = execLevelOps[op_pos];
      
      switch (op.opType) {
        case static_cast<uint8_t>(CGToucanOPName::LUT): {
          // lut
          auto op0Val = valuePool[op.lut.op0];
          auto op1Val = valuePool[op.lut.op1];
          auto op2Val = valuePool[op.lut.op2];

          uint16_t lutPos = op.lut.lutIndex + ((static_cast<uint16_t>(op0Val) << 8) | (op1Val << 4) | op2Val);
          uint8_t resultVal = lutContent[lutPos];

          auto resultPos = op.lut.result;
          valuePool[resultPos] = resultVal;
          break;
        }
        case static_cast<uint8_t>(CGToucanOPName::VecRead): {
          // vec read
          auto index0Val = static_cast<uint32_t>(valuePool[op.vec.index0]);
          auto index1Val = static_cast<uint32_t>(valuePool[op.vec.index1]);
          auto index2Val = static_cast<uint32_t>(valuePool[op.vec.index2]);
          auto index3Val = static_cast<uint32_t>(valuePool[op.vec.index3]);

          auto outRangeVal = valuePool[op.vec.outRangeValue];

          uint32_t vecOffset = ((index0Val << 12) | (index1Val << 8) | (index2Val << 4) | index3Val) + op.vec.offset;

          uint8_t resultVal = outRangeVal;

          if (vecOffset < op.vec.vecLength) {
            resultVal = valuePool[op.vec.vecBase + vecOffset];
          }
          valuePool[op.vec.result] = resultVal;
          break;
        }
        case static_cast<uint8_t>(CGToucanOPName::MemRead): {
          // mem read
          auto enVal = valuePool[op.mem.en];

          if (enVal != 0) {
            auto addrVecId = op.mem.addrVec;

            uint32_t addr = 0;
            for (size_t i = 0; i < 8; i++) {
              auto addrFragmentVal = valuePool[addrVecId + i];
              // addr = addr | (addrFragmentVal << ((7 - i) * 4));
              addr = addr | (addrFragmentVal << (i * 4));
            }

            // std::cout << "Mem depth " << op.mem.memDepth << ", addr " << addr << std::endl;
            assert(addr <= op.mem.memDepth && "Address exceed memory depth");

            if (op.mem.hasMultipleWriter) {
              assert(addr <= (UINT32_MAX >> 2) && "Memory index too large!");
              addr <<= 2;
            }
            auto realIndex = op.mem.memBase + addr;
            auto resultVal = 0;
            // This should never happen
            // assert(realIndex < design.memPool.size());
            resultVal = memPool[realIndex];
            
            valuePool[op.mem.result] = resultVal;
          } 
          // else {
          //   valuePool[op.mem.result] = 0;
          // }

          break;
        }
        default: {
          printf("Unexpected op type: %d!\n", static_cast<uint32_t>(op.opType));
          shouldStop = true;
          assert(false && "Unexpected op type!");
        }
      }
    }
  }
}

__device__ void evalLastLevel(
  uint8_t * __restrict valuePool, 
  const CGLastLevelMetaInfo * __restrict lastLevelOps, 
  const size_t numOps) {

  auto block = cg::this_thread_block();
  auto thread_rank = block.thread_rank();
  auto threads_in_block = block.size();


  for (size_t exec_pos = 0; exec_pos < numOps; exec_pos += threads_in_block) {
    size_t op_pos = exec_pos + thread_rank;
    if (op_pos < numOps) {
      // valid
      auto op = lastLevelOps[op_pos];

      switch (op.opType) {
        case static_cast<uint8_t>(CGToucanOPName::Print): {
          // print
          auto enVal = valuePool[op.print.en];
          if (enablePrint && enVal != 0) {
            auto msgPtr = printMsgs[op.print.msg];
            printf("%s", msgPtr);
          }
          break;
        }
        case static_cast<uint8_t>(CGToucanOPName::Stop): {
          // stop
          auto enVal = valuePool[op.stop.en];
          if (enVal != 0) {
            shouldStop = true;
          }
          break;
        }
        case static_cast<uint8_t>(CGToucanOPName::RegWrite): {
          // reg write
          auto datVal = valuePool[op.regWrite.dat];
          regPool[op.regWrite.reg] = datVal;
          break;
        }
        case static_cast<uint8_t>(CGToucanOPName::MemWrite): {
          // memwrite
          auto enVal = valuePool[op.memWrite.en];
          if (enVal != 0) {
            auto datVal = valuePool[op.memWrite.dat];
            auto addrVecId = op.memWrite.addrVec;

            uint32_t addr = 0;
            for (size_t i = 0; i < 8; i++) {
              auto addrFragmentVal = valuePool[addrVecId + i];
              // addr = addr | (addrFragmentVal << ((7 - i) * 4));
              addr = addr | (addrFragmentVal << (i * 4));
            }
            assert(addr <= op.memWrite.memDepth && "Address exceed memory depth");
            if (op.memWrite.hasMultipleWriter) {
              assert(addr <= (UINT32_MAX >> 2));
              addr <<= 2;
            }
            auto realIndex = op.memWrite.memBase + addr;
            memPool[realIndex] = datVal;
          }
          break;
        }
        case static_cast<uint8_t>(CGToucanOPName::ExchangeWrite): {
          // exchange write
          auto localValId = op.exgWrite.localVal;
          auto exchangeValId = op.exgWrite.exchangeVal;
          auto val = valuePool[localValId];
          exchangePool[exchangeValId] = val;
          break;
        }
        default: {
          printf("Unexpected op type: %d!\n", static_cast<uint32_t>(op.opType));
          shouldStop = true;
          assert(false && "Unexpected op type!");
        }
      }
    }
  }
}

// Note: launch in 1D
__device__ void evalEachPartition(size_t partId) {
  auto &partPtrs = partitions[partId];

  evalPartL0(partPtrs.valuePool, partPtrs.ops_l0, partPtrs.numOpsL0);
  __syncthreads();

  for (size_t exec_level_id = 0; exec_level_id < partPtrs.numExecLevels; exec_level_id++) {
    auto numOps = partPtrs.numOpsExec[exec_level_id];
    auto ops = partPtrs.ops_exec[exec_level_id];
    evalExecLevels(partPtrs.valuePool, ops, numOps);
    __syncthreads();
  }

  evalLastLevel(partPtrs.valuePool, partPtrs.ops_last, partPtrs.numOpsLast);
}

// TODO: This won't work well. 
__device__ void evalEachRegion(
  uint32_t * __restrict partIdsInCurrentRegion,
  size_t numPartsInCurrentRegion
) {
  size_t block_rank = blockIdx.x;
  size_t blocks_in_grid = blockDim.x;
  assert(blockDim.y == 1);


  for (size_t exec_pos = 0; exec_pos < numPartsInCurrentRegion; exec_pos += blocks_in_grid) {
    size_t block_pos = exec_pos + block_rank;
    if (block_pos < numPartsInCurrentRegion) {
      auto partId = partIdsInCurrentRegion[block_pos];
      evalEachPartition(partId);
    }

    return;
  }
}

__global__ void evalSingleCycle() {
  // use cooperative group
  auto grid = cg::this_grid();

  for (size_t regionId = 0; regionId < numRegions; regionId++) {
    uint32_t * partIdsInCurrentRegion = partsInRegion[regionId];
    uint32_t numPartsInCurrentRegion = numPartsInRegion[regionId];
    assert(numPartsInCurrentRegion != 0);
    evalEachRegion(partIdsInCurrentRegion, numPartsInCurrentRegion);
    __threadfence();
    grid.sync();
  }
}

__global__ void evalFreeRunningNCycles(uint32_t cycleCnt) {
  auto grid = cg::this_grid();

  for (size_t cycle = 0; cycle < cycleCnt; cycle++) {
    for (size_t regionId = 0; regionId < numRegions; regionId++) {
      uint32_t * partIdsInCurrentRegion = partsInRegion[regionId];
      uint32_t numPartsInCurrentRegion = numPartsInRegion[regionId];
      evalEachRegion(partIdsInCurrentRegion, numPartsInCurrentRegion);
      __threadfence();
      grid.sync();
    }
    grid.sync();
    if (shouldStop) {
      auto thread_rank = grid.thread_rank();
      if (thread_rank == 0) {
        realCycles = cycle + 1;
      }
      return;
    }
  }

  // update cycle counter
  auto thread_rank = grid.thread_rank();
  if (thread_rank == 0) {
    realCycles = cycleCnt;
  }
}

uint64_t read_reg_from_gpu(const std::vector<std::tuple<uint32_t, uint32_t>>& signalLocs) {
  uint64_t result;

  for(auto it = signalLocs.begin(); it != signalLocs.end(); ++it) {
    auto pos = std::get<0>(*it);
    // TODO: Use async copy to speedup
    uint8_t valFragment;
    gpuErrchk(cudaMemcpy(&valFragment, regPool_device + pos, 1, cudaMemcpyDeviceToHost));
    assert(valFragment <= 0xF);
    result = (result << 4) | valFragment;
  }
  return result;
}


void write_reg_to_gpu(const std::vector<std::tuple<uint32_t, uint32_t>>& signalLocs, uint64_t signalValue) {
  for(auto rit = signalLocs.rbegin(); rit != signalLocs.rend(); ++rit) {
    auto pos = std::get<0>(*rit);
    uint8_t valFragment = signalValue & 0xF;
    gpuErrchk(cudaMemcpyAsync(regPool_device + pos, &valFragment, 1, cudaMemcpyHostToDevice));
    signalValue = signalValue >> 4;
  }
  gpuErrchk(cudaDeviceSynchronize());
  assert(signalValue == 0 && "Given value is wider than register");
}

bool get_eval_done() {
  bool ret = false;
  cudaMemcpyFromSymbol(&ret, shouldStop, 1);
  return ret;
}

uint32_t get_real_cycles() {
  uint32_t ret;
  cudaMemcpyFromSymbol(&ret, realCycles, sizeof(uint32_t));
  return ret;
}

void setEnablePrint(bool print_en) {
  cudaMemcpyToSymbol(enablePrint, &print_en, 1);
}

void copy_netlist_to_gpu(toucanGPUSim::SimDesignInfo &design) {
  // copy lut
  assert(design.lut.size() == LUT_SIZE);
  cudaMemcpyToSymbol(lutContent, design.lut.data(), design.lut.size() * sizeof(uint8_t));

  // copy regs and mem
  assert(design.regPool.size() == design.regPoolSize && "Reg pool should be initialized!");
  assert(design.memPool.size() == design.memPoolSize && "Mem pool should be initialized!");
  assert(design.exchangePool.size() == design.exchangePoolSize && "Exchange pool should be initialized!");

  cudaMalloc(&(regPool_device), design.regPoolSize);
  cudaMalloc(&(memPool_device), design.memPoolSize);
  cudaMalloc(&(exchangePool_device), design.exchangePoolSize);
  cudaMemcpy(regPool_device, design.regPool.data(), design.regPoolSize, cudaMemcpyHostToDevice);
  cudaMemcpy(memPool_device, design.memPool.data(), design.memPoolSize, cudaMemcpyHostToDevice);
  cudaMemcpy(exchangePool_device, design.exchangePool.data(), design.exchangePoolSize, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(regPool, &regPool_device, sizeof(uint8_t*));
  cudaMemcpyToSymbol(memPool, &memPool_device, sizeof(uint8_t*));
  cudaMemcpyToSymbol(exchangePool, &exchangePool_device, sizeof(uint8_t*));
  


  // copy each partitions
  for (auto &eachPart: design.parts) {
    //
    SimPartitionPtrs partInfo;

    // copy value pool
    assert(eachPart.valuePool.size() == eachPart.valuePoolSize);
    cudaMalloc(&(partInfo.valuePool), eachPart.valuePoolSize);
    cudaMemcpy(partInfo.valuePool, eachPart.valuePool.data(), eachPart.valuePoolSize, cudaMemcpyHostToDevice);

    // copy operations

    // ops_l0
    assert(!eachPart.ops_l0.empty());
    partInfo.numOpsL0 = eachPart.ops_l0.size();
    cudaMalloc(&(partInfo.ops_l0), eachPart.ops_l0.size() * sizeof(CGTopLevelMetaInfo));
    cudaMemcpy(partInfo.ops_l0, eachPart.ops_l0.data(), eachPart.ops_l0.size() * sizeof(CGTopLevelMetaInfo), cudaMemcpyHostToDevice);

    // middle level ops
    std::vector<CGExecLevelMetaInfo*> middleLevels;
    std::vector<size_t> middleLevelSize;
    for (auto &eachExecLevel: eachPart.ops_exec) {
      assert(!eachExecLevel.empty());
      CGExecLevelMetaInfo *exec_op_ptr;
      size_t memSize = eachExecLevel.size() * sizeof(CGExecLevelMetaInfo);
      assert(memSize != 0);

      cudaMalloc(&exec_op_ptr, memSize);
      cudaMemcpy(exec_op_ptr, eachExecLevel.data(), memSize, cudaMemcpyHostToDevice);
      middleLevels.push_back(exec_op_ptr);
      middleLevelSize.push_back(eachExecLevel.size());
    }
    // Note: exec levels might be 0
    partInfo.numExecLevels = middleLevels.size();

    if (!middleLevels.empty()) {
      size_t memSize = middleLevels.size() * sizeof(CGExecLevelMetaInfo*);
      cudaMalloc(&(partInfo.ops_exec), memSize);
      cudaMemcpy(partInfo.ops_exec, middleLevels.data(), memSize, cudaMemcpyHostToDevice);

      memSize = middleLevelSize.size() * sizeof(size_t);
      cudaMalloc(&(partInfo.numOpsExec), memSize);
      cudaMemcpy(partInfo.numOpsExec, middleLevelSize.data(), memSize, cudaMemcpyHostToDevice);
    } else {
      partInfo.ops_exec = nullptr;
    }

    // ops_last
    assert(!eachPart.ops_last.empty());
    partInfo.numOpsLast = eachPart.ops_last.size();
    cudaMalloc(&(partInfo.ops_last), eachPart.ops_last.size() * sizeof(CGLastLevelMetaInfo));
    cudaMemcpy(partInfo.ops_last, eachPart.ops_last.data(), eachPart.ops_last.size() * sizeof(CGLastLevelMetaInfo), cudaMemcpyHostToDevice);

    gpuPartInfos.push_back(partInfo);
  }

  // copy partition info
  {
    SimPartitionPtrs *partInfosPtr;
    size_t memSize = gpuPartInfos.size() * sizeof(SimPartitionPtrs);
    cudaMalloc(&partInfosPtr, memSize);
    cudaMemcpy(partInfosPtr, gpuPartInfos.data(), memSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(partitions, &partInfosPtr, sizeof(SimPartitionPtrs*));
  }

  // setup numRegions and partsInRegion
  std::vector<uint32_t*> partsInRegion_device;
  std::vector<uint32_t> numParts;
  size_t _partId = 0;
  for (const auto &eachRegionParts: design.regionPartitionIds) {
    for (const auto partId: eachRegionParts) {
      assert(partId == _partId);
      _partId++;
    }
    uint32_t *ptr;
    size_t memSize = eachRegionParts.size() * sizeof(uint32_t);
    cudaMalloc(&ptr, memSize);
    cudaMemcpy(ptr, eachRegionParts.data(), memSize, cudaMemcpyHostToDevice);
    partsInRegion_device.push_back(ptr);
    numParts.push_back(eachRegionParts.size());
  }
  // Should have at least 1 region
  size_t numRegions_host = numParts.size();
  assert(numRegions_host != 0);

  uint32_t *numPartsInRegion_device;
  cudaMalloc(&numPartsInRegion_device, numRegions_host * sizeof(uint32_t*));
  cudaMemcpy(numPartsInRegion_device, numParts.data(), numRegions_host * sizeof(uint32_t*), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(numPartsInRegion, &numPartsInRegion_device, sizeof(uint32_t*));
  cudaMemcpyToSymbol(numRegions, &numRegions_host, sizeof(uint32_t));

  uint32_t **partsInRegion_device_ptrs;
  cudaMalloc(&partsInRegion_device_ptrs, numRegions_host * sizeof(uint32_t**));
  cudaMemcpy(partsInRegion_device_ptrs, partsInRegion_device.data(), numRegions_host * sizeof(uint32_t**), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(partsInRegion, &partsInRegion_device_ptrs, sizeof(uint32_t**));

  // copy print msgs
  std::vector<char*> printMsgs_device_ptrs;
  for (auto &eachMsg: design.printMsgs) {
    char* msgPtr;
    size_t memSize = (eachMsg.size() + 1) * sizeof(char);
    cudaMalloc(&msgPtr, memSize);
    cudaMemcpy(msgPtr, eachMsg.c_str(), memSize, cudaMemcpyHostToDevice);
    printMsgs_device_ptrs.push_back(msgPtr);
  }
  {
    size_t memSize = printMsgs_device_ptrs.size() * sizeof(char*);
    char **printMsgs_device;
    cudaMalloc(&(printMsgs_device), memSize);
    cudaMemcpy(printMsgs_device, printMsgs_device_ptrs.data(), memSize, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(printMsgs, &printMsgs_device, sizeof(char**));
  }

  gpuErrchk(cudaDeviceSynchronize());
}



