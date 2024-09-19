// #define NDEBUG

#include "ToucanGPUGenDataTypes.h"

#include "SimEval.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <cassert>
#include <vector>

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
#define NETLIST_ALIGNMENT 16
__host__ __device__ int getExtraAlignBytes(size_t memSize) {
  return (NETLIST_ALIGNMENT - (memSize % NETLIST_ALIGNMENT)) % NETLIST_ALIGNMENT;
};

__device__ char* align_pointer(char* ptr) {
  auto paddingSize = getExtraAlignBytes(reinterpret_cast<size_t>(ptr));
  return ptr + paddingSize;
}


#define LUT_SIZE 5154

typedef struct {
  // Private data
  uint8_t *valuePool;
  size_t valuePoolSize;
  size_t numConstsInValuePool;

  uint8_t *constVecPool;

  char* netlist;

  // Top level
  size_t numOpsL0RegRead;
  size_t numOpsL0ExgRead;

  // Exec level
  size_t *numOpsExecMemRead;
  size_t *numOpsExecVecRead;
  size_t *numOpsExecLUT;

  // Last level
  size_t numOpsLastExgWrite;
  size_t numOpsLastRegWrite;
  size_t numOpsLastMemWrite;
  size_t numOpsLastPrint;
  size_t numOpsLastStop;

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


extern __shared__ uint8_t sharedMem[];

// every thread group
__device__ void evalPartL0(
  uint8_t * __restrict valuePool, 
  const toucanGPUSim::CGRegReadMetaInfo * __restrict topLevelRegReadOps,
  const toucanGPUSim::CGExchangeReadMetaInfo * __restrict topLevelExgReadOps,
  const size_t numRegReads,
  const size_t numExgReads) {

  auto block = cg::this_thread_block();
  auto thread_rank = block.thread_rank();
  auto threads_in_block = block.size();

  // Eval reg reads
  for (size_t op_pos = thread_rank; op_pos < numRegReads; op_pos += threads_in_block) {
    // reg read
    const auto &op = topLevelRegReadOps[op_pos];
    auto regValId = op.reg;
    auto resultId = op.result;
    auto regVal = regPool[regValId];
    valuePool[resultId] = regVal;
  }

  // Eval exchange reads
  for (size_t op_pos = thread_rank; op_pos < numExgReads; op_pos += threads_in_block) {
    const auto &op = topLevelExgReadOps[op_pos];
    auto exgValId = op.exchangeVal;
    auto localValId = op.localVal;
    auto exgVal = exchangePool[exgValId];
    valuePool[localValId] = exgVal;
  }
}

__device__ void evalExecLevels(
  const uint8_t * __restrict constVecPool,
  uint8_t * __restrict valuePool, 
  const toucanGPUSim::CGMemReadMetaInfo * __restrict memReadOps,
  const toucanGPUSim::CGVecReadMetaInfo * __restrict vecReadOps,
  const toucanGPUSim::CGLUTMetaInfo * __restrict lutOps,
  const size_t numMemReadOps,
  const size_t numVecReadOps,
  const size_t numLUTOps) {

  auto block = cg::this_thread_block();
  auto thread_rank = block.thread_rank();
  auto threads_in_block = block.size();

  for (size_t op_pos = thread_rank; op_pos < numMemReadOps; op_pos += threads_in_block) {
    const auto op = memReadOps[op_pos];

    auto enVal = valuePool[op.en];
    if (enVal != 0) {
      auto addrVecId = op.addrVec;

      uint32_t addr = 0;
      for (size_t i = 0; i < 8; i++) {
        auto addrFragmentVal = valuePool[addrVecId + i];
        addr = addr | (addrFragmentVal << (i * 4));
      }

      // assert(addr <= op.memDepth && "Address exceed memory depth");
      if (op.hasMultipleWriter) {
        // assert(addr <= (UINT32_MAX >> 2) && "Memory index too large!");
        addr <<= 2;
      }
      auto realIndex = op.memBase + addr;
      auto resultVal = memPool[realIndex];
      
      valuePool[op.result] = resultVal;
    } 
    // else {
    //   valuePool[op.mem.result] = 0;
    // }
  }

  for (size_t op_pos = thread_rank; op_pos < numVecReadOps; op_pos += threads_in_block) {
    // vec read
    const auto op = vecReadOps[op_pos];

    auto isConstVec = op.isConstVec;
        
    auto index0Val = static_cast<uint32_t>(valuePool[op.index0]);
    auto index1Val = static_cast<uint32_t>(valuePool[op.index1]);
    auto index2Val = static_cast<uint32_t>(valuePool[op.index2]);
    auto index3Val = static_cast<uint32_t>(valuePool[op.index3]);

    auto outRangeVal = valuePool[op.outRangeValue];

    uint32_t vecOffset = ((index0Val << 12) | (index1Val << 8) | (index2Val << 4) | index3Val) + op.offset;

    uint8_t resultVal = outRangeVal;

    if (vecOffset < op.vecLength) {
      if (isConstVec) {
        resultVal = constVecPool[op.vecBase + vecOffset];
      } else {
        resultVal = valuePool[op.vecBase + vecOffset];
      }
    }
    valuePool[op.result] = resultVal;
  }

  for (size_t op_pos = thread_rank; op_pos < numLUTOps; op_pos += threads_in_block) {
    const auto op = lutOps[op_pos];

    auto op0Val = valuePool[op.op0];
    auto op1Val = valuePool[op.op1];
    auto op2Val = valuePool[op.op2];

    uint16_t lutPos = op.lutIndex + ((static_cast<uint16_t>(op0Val) << 8) | (op1Val << 4) | op2Val);
    uint8_t resultVal = lutContent[lutPos];

    auto resultPos = op.result;
    valuePool[resultPos] = resultVal;
  }

}

__device__ void evalLastLevel(
  uint8_t * __restrict valuePool, 
  const toucanGPUSim::CGExchangeWriteMetaInfo * __restrict exgWriteOps,
  const toucanGPUSim::CGRegWriteMetaInfo * __restrict regWriteOps,
  const toucanGPUSim::CGMemWriteMetaInfo * __restrict memWriteOps,
  const toucanGPUSim::CGPrintMetaInfo * __restrict printOps,
  const toucanGPUSim::CGStopMetaInfo * __restrict stopOps,
  const size_t numExgWriteOps,
  const size_t numRegWriteOps,
  const size_t numMemWriteOps,
  const size_t numPrintOps,
  const size_t numStopOps) {

  auto block = cg::this_thread_block();
  auto thread_rank = block.thread_rank();
  auto threads_in_block = block.size();

  for (size_t op_pos = thread_rank; op_pos < numMemWriteOps; op_pos += threads_in_block) {
    const auto op = memWriteOps[op_pos];

    // memwrite
    auto enVal = valuePool[op.en];
    if (enVal != 0) {
      auto datVal = valuePool[op.dat];
      auto addrVecId = op.addrVec;

      uint32_t addr = 0;
      for (size_t i = 0; i < 8; i++) {
        auto addrFragmentVal = valuePool[addrVecId + i];
        addr = addr | (addrFragmentVal << (i * 4));
      }
      // assert(addr <= op.memDepth && "Address exceed memory depth");
      if (op.hasMultipleWriter) {
        // assert(addr <= (UINT32_MAX >> 2));
        addr <<= 2;
      }
      auto realIndex = op.memBase + addr;
      memPool[realIndex] = datVal;
    }
  }

  for (size_t op_pos = thread_rank; op_pos < numExgWriteOps; op_pos += threads_in_block) {
    const auto op = exgWriteOps[op_pos];

    // exchange write
    auto localValId = op.localVal;
    auto exchangeValId = op.exchangeVal;
    auto val = valuePool[localValId];
    exchangePool[exchangeValId] = val;
  }

  for (size_t op_pos = thread_rank; op_pos < numRegWriteOps; op_pos += threads_in_block) {
    const auto op = regWriteOps[op_pos];

    // reg write
    auto datVal = valuePool[op.dat];
    regPool[op.reg] = datVal;
  }

  for (size_t op_pos = thread_rank; op_pos < numPrintOps; op_pos += threads_in_block) {
    const auto op = printOps[op_pos];
    auto enVal = valuePool[op.en];
    if (enablePrint && enVal != 0) {
      auto msgPtr = printMsgs[op.msg];
      printf("%s", msgPtr);
    }
  }

  for (size_t op_pos = thread_rank; op_pos < numStopOps; op_pos += threads_in_block) {
    const auto op = stopOps[op_pos];

    // stop
    auto enVal = valuePool[op.en];
    if (enVal != 0) {
      shouldStop = true;
    }
  }

}

// Note: launch in 1D
__device__ void evalEachPartition(size_t partId) {
  auto &partPtrs = partitions[partId];

  auto block = cg::this_thread_block();
  auto thread_rank = block.thread_rank();
  auto threads_in_block = block.size();

  uint8_t *localValuePool = reinterpret_cast<uint8_t*>(sharedMem);
  // load consts
  for (size_t data_pos = thread_rank; data_pos < partPtrs.numConstsInValuePool; data_pos += threads_in_block) {
    localValuePool[data_pos] = partPtrs.valuePool[data_pos];
  }

  auto netlist_ptr = partPtrs.netlist;
  // char* netlist_eval_start;

  auto netlist_exgRead = align_pointer(netlist_ptr);
  auto netlist_regRead = align_pointer(netlist_ptr + (partPtrs.numOpsL0ExgRead * sizeof(toucanGPUSim::CGExchangeReadMetaInfo)));

  evalPartL0(localValuePool, 
  reinterpret_cast<const toucanGPUSim::CGRegReadMetaInfo*>(netlist_regRead), 
  reinterpret_cast<const toucanGPUSim::CGExchangeReadMetaInfo*>(netlist_exgRead), 
  partPtrs.numOpsL0RegRead, partPtrs.numOpsL0ExgRead);
  __syncthreads();

  auto netlist_eval_start = netlist_regRead + (partPtrs.numOpsL0RegRead * sizeof(toucanGPUSim::CGRegReadMetaInfo));

  for (size_t exec_level_id = 0; exec_level_id < partPtrs.numExecLevels; exec_level_id++) {
    auto num_memRead = partPtrs.numOpsExecMemRead[exec_level_id];
    auto num_vecRead = partPtrs.numOpsExecVecRead[exec_level_id];
    auto num_lut = partPtrs.numOpsExecLUT[exec_level_id];

    auto netlist_memRead = align_pointer(netlist_eval_start);
    auto netlist_vecRead = align_pointer(netlist_memRead + (num_memRead * sizeof(toucanGPUSim::CGMemReadMetaInfo)));
    auto netlist_lut = align_pointer(netlist_vecRead + (num_vecRead * sizeof(toucanGPUSim::CGVecReadMetaInfo)));
    evalExecLevels(
      partPtrs.constVecPool,
      localValuePool, 
      reinterpret_cast<const toucanGPUSim::CGMemReadMetaInfo*>(netlist_memRead),
      reinterpret_cast<const toucanGPUSim::CGVecReadMetaInfo*>(netlist_vecRead), 
      reinterpret_cast<const toucanGPUSim::CGLUTMetaInfo*>(netlist_lut), 
      num_memRead, 
      num_vecRead, 
      num_lut);
    netlist_eval_start = align_pointer(netlist_lut + (num_lut * sizeof(toucanGPUSim::CGLUTMetaInfo)));

    __syncthreads();
  }

  auto num_exgWrite = partPtrs.numOpsLastExgWrite;
  auto num_regWrite = partPtrs.numOpsLastRegWrite;
  auto num_memWrite = partPtrs.numOpsLastMemWrite;
  auto num_print = partPtrs.numOpsLastPrint;
  auto num_stop = partPtrs.numOpsLastStop;

  auto netlist_exgWrite = align_pointer(netlist_eval_start);
  auto netlist_regWrite = align_pointer(netlist_exgWrite + (num_exgWrite * sizeof(toucanGPUSim::CGExchangeWriteMetaInfo)));
  auto netlist_memWrite = align_pointer(netlist_regWrite + (num_regWrite * sizeof(toucanGPUSim::CGRegWriteMetaInfo)));
  auto netlist_print = align_pointer(netlist_memWrite + (num_memWrite * sizeof(toucanGPUSim::CGMemWriteMetaInfo)));
  auto netlist_stop = align_pointer(netlist_print + (num_print * sizeof(toucanGPUSim::CGPrintMetaInfo)));

  evalLastLevel(
    localValuePool, 
    reinterpret_cast<const toucanGPUSim::CGExchangeWriteMetaInfo*>(netlist_exgWrite), 
    reinterpret_cast<const toucanGPUSim::CGRegWriteMetaInfo*>(netlist_regWrite), 
    reinterpret_cast<const toucanGPUSim::CGMemWriteMetaInfo*>(netlist_memWrite), 
    reinterpret_cast<const toucanGPUSim::CGPrintMetaInfo*>(netlist_print), 
    reinterpret_cast<const toucanGPUSim::CGStopMetaInfo*>(netlist_stop), 
    num_exgWrite, 
    num_regWrite, 
    num_memWrite, 
    num_print, 
    num_stop);

}


__device__ void evalEachRegion(
  const uint32_t * __restrict partIdsInCurrentRegion,
  const size_t numPartsInCurrentRegion
) {
  size_t block_rank = blockIdx.x;
  size_t blocks_in_grid = blockDim.x;

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
  uint64_t result = 0;

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

template <typename T>
static void allocAndCopyVector(T **devicePtr, const void *data, const size_t size) {
  cudaMalloc(devicePtr, size);
  cudaMemcpy(*devicePtr, data, size, cudaMemcpyHostToDevice);
}

void copy_netlist_to_gpu(toucanGPUSim::SimDesignInfo &design) {
  // copy lut
  assert(design.lut.size() == LUT_SIZE);
  cudaMemcpyToSymbol(lutContent, design.lut.data(), design.lut.size() * sizeof(uint8_t));

  // copy regs and mem
  assert(design.regPool.size() == design.regPoolSize && "Reg pool should be initialized!");
  assert(design.memPool.size() == design.memPoolSize && "Mem pool should be initialized!");
  assert(design.exchangePool.size() == design.exchangePoolSize && "Exchange pool should be initialized!");

  allocAndCopyVector(&regPool_device, design.regPool.data(), design.regPoolSize);
  allocAndCopyVector(&memPool_device, design.memPool.data(), design.memPoolSize);
  allocAndCopyVector(&exchangePool_device, design.exchangePool.data(), design.exchangePoolSize);
  
  cudaMemcpyToSymbol(regPool, &regPool_device, sizeof(uint8_t*));
  cudaMemcpyToSymbol(memPool, &memPool_device, sizeof(uint8_t*));
  cudaMemcpyToSymbol(exchangePool, &exchangePool_device, sizeof(uint8_t*));
  


  // copy each partitions
  for (auto &eachPart: design.parts) {
    //
    SimPartitionPtrs partInfo;
    std::vector<char> allNetlist;

    auto appendToNetlistVec = [&](const char * dat, size_t numBytes) {
      auto alignBytes = getExtraAlignBytes(numBytes);
      allNetlist.insert(allNetlist.end(), dat, dat + numBytes);
      for (size_t i = 0; i < alignBytes; i++) {
        allNetlist.push_back(0);
      }
    };

    // copy value pool
    assert(eachPart.valuePool.size() == eachPart.valuePoolSize);
    allocAndCopyVector(&(partInfo.valuePool), eachPart.valuePool.data(), eachPart.valuePoolSize);
    partInfo.valuePoolSize = eachPart.valuePoolSize;
    assert(partInfo.valuePoolSize <= UINT16_MAX);
    partInfo.numConstsInValuePool = eachPart.numConstsInValuePool;

    // copy const vec pool
    if (eachPart.constVecPool.empty()) {
      partInfo.constVecPool = nullptr;
    } else {
      allocAndCopyVector(&(partInfo.constVecPool), eachPart.constVecPool.data(), eachPart.constVecPool.size());
    }

    // copy operations

    // ops_l0
    partInfo.numOpsL0ExgRead = eachPart.ops_l0_exgRead.size();
    partInfo.numOpsL0RegRead = eachPart.ops_l0_regRead.size();
    assert((partInfo.numOpsL0ExgRead == 0) || (partInfo.numOpsL0RegRead == 0));

    if (partInfo.numOpsL0ExgRead != 0) {
      size_t memSize = eachPart.ops_l0_exgRead.size() * sizeof(toucanGPUSim::CGExchangeReadMetaInfo);
      appendToNetlistVec(reinterpret_cast<const char*>(eachPart.ops_l0_exgRead.data()), memSize);
    }

    if (partInfo.numOpsL0RegRead != 0) {
      size_t memSize = eachPart.ops_l0_regRead.size() * sizeof(toucanGPUSim::CGRegReadMetaInfo);
      appendToNetlistVec(reinterpret_cast<const char*>(eachPart.ops_l0_regRead.data()), memSize);
    }


    // middle level ops
    std::vector<size_t> exec_levels_numMemReads;
    std::vector<size_t> exec_levels_numVecReads;
    std::vector<size_t> exec_levels_numLUTs;

    size_t numExecLevels = eachPart.ops_exec_memRead.size();
    assert(eachPart.ops_exec_vecRead.size() == numExecLevels);
    assert(eachPart.ops_exec_lut.size() == numExecLevels);

    for (size_t level_id = 0; level_id < eachPart.ops_exec_memRead.size(); level_id++) {
      const auto &part_memRead = eachPart.ops_exec_memRead[level_id];
      const auto &part_vecRead = eachPart.ops_exec_vecRead[level_id];
      const auto &part_lut = eachPart.ops_exec_lut[level_id];

      exec_levels_numMemReads.push_back(part_memRead.size());
      exec_levels_numVecReads.push_back(part_vecRead.size());
      exec_levels_numLUTs.push_back(part_lut.size());

      if (!part_memRead.empty()) {
        size_t memSize = part_memRead.size() * sizeof(toucanGPUSim::CGMemReadMetaInfo);
        appendToNetlistVec(reinterpret_cast<const char*>(part_memRead.data()), memSize);
      }

      if (!part_vecRead.empty()) {
        toucanGPUSim::CGVecReadMetaInfo *op_ptr;
        size_t memSize = part_vecRead.size() * sizeof(toucanGPUSim::CGVecReadMetaInfo);
        allocAndCopyVector(&op_ptr, part_vecRead.data(), memSize);
        appendToNetlistVec(reinterpret_cast<const char*>(part_vecRead.data()), memSize);
      }

      if (!part_lut.empty()) {
        size_t memSize = part_lut.size() * sizeof(toucanGPUSim::CGLUTMetaInfo);
        appendToNetlistVec(reinterpret_cast<const char*>(part_lut.data()), memSize);
      }
    }

    partInfo.numExecLevels = numExecLevels;

    if (numExecLevels != 0) {
      size_t memSize = 0;

      // memRead counts
      memSize = exec_levels_numMemReads.size() * sizeof(size_t);
      allocAndCopyVector(&(partInfo.numOpsExecMemRead), exec_levels_numMemReads.data(), memSize);

      // counts
      memSize = exec_levels_numVecReads.size() * sizeof(size_t);
      allocAndCopyVector(&(partInfo.numOpsExecVecRead), exec_levels_numVecReads.data(), memSize);

      // luts
      memSize = exec_levels_numLUTs.size() * sizeof(size_t);
      allocAndCopyVector(&(partInfo.numOpsExecLUT), exec_levels_numLUTs.data(), memSize);
    } else {
      // no exec levels
      partInfo.numOpsExecMemRead = nullptr;
      partInfo.numOpsExecVecRead = nullptr;
      partInfo.numOpsExecLUT = nullptr;
    }

    // last level
    partInfo.numOpsLastExgWrite = eachPart.ops_last_exgWrite.size();
    partInfo.numOpsLastRegWrite = eachPart.ops_last_regWrite.size();
    partInfo.numOpsLastMemWrite = eachPart.ops_last_memWrite.size();
    partInfo.numOpsLastPrint = eachPart.ops_last_print.size();
    partInfo.numOpsLastStop = eachPart.ops_last_stop.size();

    if (partInfo.numOpsLastExgWrite != 0) {
      size_t memSize = eachPart.ops_last_exgWrite.size() * sizeof(toucanGPUSim::CGExchangeWriteMetaInfo);
      appendToNetlistVec(reinterpret_cast<const char*>(eachPart.ops_last_exgWrite.data()), memSize);
    }

    if (partInfo.numOpsLastRegWrite != 0) {
      size_t memSize = eachPart.ops_last_regWrite.size() * sizeof(toucanGPUSim::CGRegWriteMetaInfo);
      appendToNetlistVec(reinterpret_cast<const char*>(eachPart.ops_last_regWrite.data()), memSize);
    }

    if (partInfo.numOpsLastMemWrite != 0) {
      size_t memSize = eachPart.ops_last_memWrite.size() * sizeof(toucanGPUSim::CGMemWriteMetaInfo);
      appendToNetlistVec(reinterpret_cast<const char*>(eachPart.ops_last_memWrite.data()), memSize);
    }

    if (partInfo.numOpsLastPrint != 0) {
      size_t memSize = eachPart.ops_last_print.size() * sizeof(toucanGPUSim::CGPrintMetaInfo);
      appendToNetlistVec(reinterpret_cast<const char*>(eachPart.ops_last_print.data()), memSize);
    }

    if (partInfo.numOpsLastStop != 0) {
      size_t memSize = eachPart.ops_last_stop.size() * sizeof(toucanGPUSim::CGStopMetaInfo);
      appendToNetlistVec(reinterpret_cast<const char*>(eachPart.ops_last_stop.data()), memSize);
    }

    allocAndCopyVector(&(partInfo.netlist), allNetlist.data(), allNetlist.size() * sizeof(char));

    gpuPartInfos.push_back(partInfo);
  }

  // copy partition info
  {
    SimPartitionPtrs *partInfosPtr;
    size_t memSize = gpuPartInfos.size() * sizeof(SimPartitionPtrs);
    allocAndCopyVector(&partInfosPtr, gpuPartInfos.data(), memSize);
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
    allocAndCopyVector(&ptr, eachRegionParts.data(), memSize);
    partsInRegion_device.push_back(ptr);
    numParts.push_back(eachRegionParts.size());
  }
  // Should have at least 1 region
  size_t numRegions_host = numParts.size();
  assert(numRegions_host != 0);

  uint32_t *numPartsInRegion_device;
  allocAndCopyVector(&numPartsInRegion_device, numParts.data(), numRegions_host * sizeof(uint32_t*));
  cudaMemcpyToSymbol(numPartsInRegion, &numPartsInRegion_device, sizeof(uint32_t*));
  cudaMemcpyToSymbol(numRegions, &numRegions_host, sizeof(uint32_t));

  uint32_t **partsInRegion_device_ptrs;
  allocAndCopyVector(&partsInRegion_device_ptrs, partsInRegion_device.data(), numRegions_host * sizeof(uint32_t**));
  cudaMemcpyToSymbol(partsInRegion, &partsInRegion_device_ptrs, sizeof(uint32_t**));

  // copy print msgs
  std::vector<char*> printMsgs_device_ptrs;
  for (auto &eachMsg: design.printMsgs) {
    char* msgPtr;
    size_t memSize = (eachMsg.size() + 1) * sizeof(char);
    allocAndCopyVector(&msgPtr, eachMsg.c_str(), memSize);
    printMsgs_device_ptrs.push_back(msgPtr);
  }
  {
    size_t memSize = printMsgs_device_ptrs.size() * sizeof(char*);
    char **printMsgs_device;
    allocAndCopyVector(&(printMsgs_device), printMsgs_device_ptrs.data(), memSize);
    cudaMemcpyToSymbol(printMsgs, &printMsgs_device, sizeof(char**));
  }

  gpuErrchk(cudaDeviceSynchronize());
}



