// #define NDEBUG

// Note: Enable async memcpy for reg write by uncomment this macro
// Seems not helpful for performance
// #define REG_WRITE_USE_ASYNC_MEMCPY

#include "ToucanGPUGenDataTypes.h"

#include "SimEval.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <cassert>
#include <vector>

#include <cooperative_groups.h>

#ifdef REG_WRITE_USE_ASYNC_MEMCPY
#include <cooperative_groups/memcpy_async.h>
// #include <cuda/barrier>
#endif

using namespace toucanGPUSim;
namespace cg = cooperative_groups;





#define NETLIST_ALIGNMENT 16
#define LUT_SIZE 5154

#define MICROPART_REGULAR_MAX_LEVELS 64
#define MICROPART_REGULAR_MAGIC 0xfb709394
#define MICROPART_VECREAD_MAGIC 0xfb809917
#define MICROPART_MEMREAD_MAGIC 0xcccccccc
#define MICROPART_VECOP_MAGIC 0xbeef1234

#define LUT_NOP_INDEX 0


__host__ __device__ size_t getExtraAlignBytes(size_t memSize) {
  return (NETLIST_ALIGNMENT - (memSize % NETLIST_ALIGNMENT)) % NETLIST_ALIGNMENT;
};

__device__ char* align_pointer(char* ptr) {
  auto paddingSize = getExtraAlignBytes(reinterpret_cast<size_t>(ptr));
  return ptr + paddingSize;
}

__host__ __device__ size_t getNumPadWithExtraAlignment(size_t elementSize, size_t alignment) {
  size_t extraPadding = alignment - (elementSize % alignment);
  if (extraPadding == alignment) extraPadding = 0;
  return elementSize + extraPadding;
};



typedef struct {
  uint32_t magic;
  uint32_t partMiddleLevels;
  uint8_t numOpsAtEachLevel[MICROPART_REGULAR_MAX_LEVELS];
} SimMicroPartNetlistHeader;



void copyMicroPartToNetlist(const toucanGPUSim::CGMicroPartInfo &mPart, std::vector<char> &bytes) {
  auto appendUINT32 = [&bytes](uint32_t dat) {
    for (size_t i = 0; i < sizeof(uint32_t); i++) {
      bytes.push_back(reinterpret_cast<char*>(&dat)[i]);
    }
  };

  auto appendByte = [&bytes](uint8_t dat) {
    bytes.push_back(static_cast<char>(dat));
  };

  auto appendOperations = [&bytes](const auto& operations) {
    const char* data = reinterpret_cast<const char*>(operations.data());
    size_t size = operations.size() * sizeof(typename std::decay<decltype(operations)>::type::value_type);
    bytes.insert(bytes.end(), data, data + size);
  };

  if (mPart.isLUTPart) {
    // LUT part serialization
    appendUINT32(MICROPART_REGULAR_MAGIC);
    appendUINT32(mPart.middleLevels.size());
    
    // Validate constraints
    assert(mPart.middleLevels.size() + 2 <= MICROPART_REGULAR_MAX_LEVELS);
    assert(mPart.topLevel.size() <= 32);
    assert(mPart.lastLevel.size() <= 32);
    for (const auto &eachLevel: mPart.middleLevels) {
      assert(eachLevel.size() <= 32);
    }

    // Write level sizes header (MICROPART_REGULAR_MAX_LEVELS bytes)
    // topLevel size
    appendByte(static_cast<uint8_t>(mPart.topLevel.size()));
    
    // middleLevels sizes
    for (const auto &eachLevel: mPart.middleLevels) {
      appendByte(static_cast<uint8_t>(eachLevel.size()));
    }
    
    // lastLevel size
    appendByte(static_cast<uint8_t>(mPart.lastLevel.size()));
    
    // Pad remaining bytes to reach MICROPART_REGULAR_MAX_LEVELS
    size_t usedBytes = 1 + mPart.middleLevels.size() + 1; // topLevel + middleLevels + lastLevel
    for (size_t i = usedBytes; i < MICROPART_REGULAR_MAX_LEVELS; i++) {
      appendByte(0);
    }

    // Serialize operations in order
    // 1. topLevel operations
    appendOperations(mPart.topLevel);
    
    // 2. middleLevels operations (each level)
    for (const auto &eachLevel: mPart.middleLevels) {
      appendOperations(eachLevel);
    }
    
    // 3. lastLevel operations
    appendOperations(mPart.lastLevel);

  } else if (mPart.vecRead.size() != 0) {
    // Vector read part
    appendUINT32(MICROPART_VECREAD_MAGIC);
    appendUINT32(mPart.vecRead.size());
    appendOperations(mPart.vecRead);
    
  } else if (mPart.vecArithAndLogic.size() != 0) {
    // Vector arithmetic/logic part
    appendUINT32(MICROPART_VECOP_MAGIC);
    appendUINT32(mPart.vecArithAndLogic.size());
    appendOperations(mPart.vecArithAndLogic);
    
  } else {
    // Memory read part
    assert(mPart.memRead.size() != 0);
    appendUINT32(MICROPART_MEMREAD_MAGIC);
    appendUINT32(mPart.memRead.size());
    appendOperations(mPart.memRead);
  }
}




// Device function to evaluate a single MicroPart
// Each MicroPart should be evaluated by a single GPU thread warp
__device__ void evalSingleMicroPart(
  uint8_t * __restrict valuePool,
  const uint8_t * __restrict constVecPool,
  char * __restrict netlistPtr
) {
  auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
  auto lane_id = warp.thread_rank();
  
  // Read the header to determine MicroPart type
  uint32_t *header = reinterpret_cast<uint32_t*>(netlistPtr);
  auto magic = header[0];
  auto second_uint = header[1];

  
  
  if (magic == MICROPART_REGULAR_MAGIC) {
    // LUT MicroPart
    SimMicroPartNetlistHeader *header = reinterpret_cast<SimMicroPartNetlistHeader*>(netlistPtr);

    char *dataPtr = netlistPtr + sizeof(SimMicroPartNetlistHeader);
    
    uint32_t numMiddleLevels = second_uint;
    uint8_t *levelSizes = header->numOpsAtEachLevel;
    
    // Get level sizes
    // uint8_t topLevelSize = levelSizes[0];
    // uint8_t lastLevelSize = levelSizes[1 + numMiddleLevels];
    
    // Process operations sequentially through levels
    char *currentPtr = dataPtr;
    uint8_t thisLaneShuffleVal = 0;
    
    // 1. Process top level operations (each thread handles one operation)
    uint8_t currentLevelSize = levelSizes[0];
    if (lane_id < currentLevelSize) {
      auto topOps = reinterpret_cast<const toucanGPUSim::CGMicroPartLUTTopLevelOp*>(currentPtr);
      const auto &op = topOps[lane_id];
      auto lutIndex = op.lutIndex;
      auto op0 = op.op0;
      auto op1 = op.op1;
      auto op2 = op.op2;
      auto op0Val = (op0 < 16) ? op0 : valuePool[op0];
      auto op1Val = (op1 < 16) ? op1 : valuePool[op1];
      auto op2Val = (op2 < 16) ? op2 : valuePool[op2];
      
      uint16_t lutPos = op.lutIndex + ((static_cast<uint16_t>(op0Val) << 8) | (op1Val << 4) | op2Val);
      uint8_t resultVal = lutContent[lutPos];

      thisLaneShuffleVal = resultVal;
    }
    currentPtr += currentLevelSize * sizeof(toucanGPUSim::CGMicroPartLUTTopLevelOp);
    
    warp.sync();
    
    // 2. Process middle level operations
    for (uint32_t level = 0; level < numMiddleLevels; level++) {
      currentLevelSize = levelSizes[1 + level];

      auto middleOps = reinterpret_cast<const toucanGPUSim::CGMicroPartLUTMiddleLevelOp*>(currentPtr);
      const auto &op = middleOps[lane_id];

      // Default value. Can be changed to any lane
      uint16_t lutIndex = LUT_NOP_INDEX;
      uint8_t op0 = 0;
      uint8_t op1 = 0;
      uint8_t op2 = 0;

      if (lane_id < currentLevelSize) {
        // Decode operands (0~15: const, 32~63: value from other threads)
        lutIndex = op.lutIndex();
        auto op0 = op.op0();
        auto op1 = op.op1();
        auto op2 = op.op2();
      }

      // Every thread in the wrap should participate in shuffle
      // T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
      auto op0Val = __shfl_sync(0xFFFFFFFF, thisLaneShuffleVal, op0);
      auto op1Val = __shfl_sync(0xFFFFFFFF, thisLaneShuffleVal, op1);
      auto op2Val = __shfl_sync(0xFFFFFFFF, thisLaneShuffleVal, op2);

      // update value if it's actually a const
      if (op0 < 32) {
        assert(op0 < 16);
        op0Val = op0;
      }
      if (op1 < 32) {
        assert(op1 < 16);
        op1Val = op1;
      }
      if (op2 < 32) {
        assert(op2 < 16);
        op2Val = op2;
      }

      uint8_t resultVal;
      if (lutIndex == LUT_NOP_INDEX) {
        resultVal = op2Val;
      } else {
        uint16_t lutPos = op.lutIndex() + ((static_cast<uint16_t>(op0Val) << 8) | (op1Val << 4) | op2Val);
        resultVal = lutContent[lutPos];
      }
      
      // store for next shuffle
      thisLaneShuffleVal = resultVal;

      currentPtr += currentLevelSize * sizeof(toucanGPUSim::CGMicroPartLUTMiddleLevelOp);
      warp.sync();
    }
    
    // 3. Process last level operations (write back)
    currentLevelSize = levelSizes[1 + numMiddleLevels];

    auto lastOps = reinterpret_cast<const toucanGPUSim::CGMicroPartLUTLastLevelWriteBack*>(currentPtr);
    const auto &op = lastOps[lane_id];

    uint8_t shuffleId = 0;
    uint16_t resultId = 0;

    if (lane_id < currentLevelSize) {
      shuffleId = op.shuffleId;
      resultId = op.result;
    }

    auto resultVal = __shfl_sync(0xFFFFFFFF, thisLaneShuffleVal, shuffleId);

    // update value if it's actually a const
    if (shuffleId < 32) {
      assert(shuffleId < 16);
      resultVal = shuffleId;
    }

    if (lane_id < currentLevelSize) {
      // write back to smem
      valuePool[resultId] = resultVal;
    }
    
  } else if (magic == MICROPART_VECREAD_MAGIC) {
    // Vector read MicroPart
    uint32_t numOps = second_uint;
    char *dataPtr = netlistPtr + (2 * sizeof(uint32_t));
    
    if (lane_id < numOps) {
      auto vecReadOps = reinterpret_cast<const toucanGPUSim::CGMicroPartVecRead*>(dataPtr);
      const auto &op = vecReadOps[lane_id];
      
      auto index0Val = static_cast<uint32_t>(valuePool[op.index0]);
      auto index1Val = static_cast<uint32_t>(valuePool[op.index1]);
      auto index2Val = static_cast<uint32_t>(valuePool[op.index2]);
      auto index3Val = static_cast<uint32_t>(valuePool[op.index3]);
      
      
      
      uint32_t vecOffset = ((index0Val << 12) | (index1Val << 8) | (index2Val << 4) | index3Val) + op.offset;
      
      uint8_t resultVal;

      if (vecOffset < op.vecLength) {
        if (op.isConstVec) {
          resultVal = constVecPool[op.vecBase + vecOffset];
        } else {
          resultVal = valuePool[op.vecBase + vecOffset];
        }
      } else {
        auto outRangeVal = valuePool[op.outRangeValue];
        resultVal = outRangeVal;
      }
      valuePool[op.result] = resultVal;
    }
    
  } else if (magic == MICROPART_VECOP_MAGIC) {
    // Vector arithmetic/logic MicroPart
    uint32_t numOps = second_uint;
    char *dataPtr = netlistPtr + (2 * sizeof(uint32_t));
    
    if (lane_id < numOps) {
      auto vecOpOps = reinterpret_cast<const toucanGPUSim::CGMicroPartVecArithOrLogic*>(dataPtr);
      const auto &op = vecOpOps[lane_id];

      // For now, only support vector element width 4
      const auto vecElemWidth = 4;
      auto vecLength = op.vecLength;
      
      // Process vector operations
      assert(vecLength * 4 < 128);

      bool isV1Const = (op.isV1V2Const & 0b10) != 0;
      bool isV2Const = (op.isV1V2Const & 0b01) != 0;

      auto v1Base = op.vec1Base;
      auto v2Base = op.vec2Base;

      __int128 v1Val, v2Val;

      for (int i = 0; i < vecLength; i++) {
        __int128 temp;

        uint8_t v1_seg = isV1Const ? constVecPool[v1Base + i] : valuePool[v1Base + i];
        uint8_t v2_seg = isV2Const ? constVecPool[v2Base + i] : valuePool[v2Base + i];

        temp = v1_seg;
        v1Val = v1Val | (temp << (i * vecElemWidth));

        temp = v2_seg;
        v2Val = v2Val | (temp << (i * vecElemWidth));
      }

      const auto &opName = op.opName;
      bool resultIsVec = (opName == VEC_ARITH_ADD) || (opName == VEC_ARITH_SUB) || (opName == VEC_ARITH_MUL);

      uint8_t resultVal = 0;
      __int128 v1Result = v1Val;
      switch (opName) {
        case VEC_ARITH_ADD: v1Result = (v1Val + v2Val); break;
        case VEC_ARITH_SUB: v1Result = (v1Val - v2Val); break;
        case VEC_ARITH_MUL: v1Result = (v1Val * v2Val); break;
        case VEC_LOGIC_EQ: resultVal = (v1Val == v2Val) ? 1 : 0; break;
        case VEC_LOGIC_LT: resultVal = (v1Val < v2Val) ? 1 : 0; break;
        case VEC_LOGIC_LE: resultVal = (v1Val <= v2Val) ? 1 : 0; break;
      }

      if (resultIsVec) {
        for (int i = 0; i < vecLength; i++) {
          uint8_t seg = v1Result & 0xF;
          uint16_t resultId = op.result + i;
          valuePool[resultId] = seg;
          v1Result = v1Result >> 4;
        }
      } else {
        valuePool[op.result] = resultVal;
      }

    }

  } else if (magic == MICROPART_MEMREAD_MAGIC) {
    // Memory read MicroPart
    uint32_t numOps = second_uint;
    char *dataPtr = netlistPtr + (2 * sizeof(uint32_t));
    
    if (lane_id < numOps) {
      auto memReadOps = reinterpret_cast<const toucanGPUSim::CGMicroPartMemRead*>(dataPtr);
      const auto &op = memReadOps[lane_id];
      
      auto enVal = valuePool[op.en];
      if (enVal != 0) {
        uint32_t addr = 0;
        for (size_t j = 0; j < 8; j++) {
          auto addrFragmentVal = valuePool[op.addrVec + j];
          addr = addr | (addrFragmentVal << (j * 4));
        }
        
        if (op.hasMultipleWriter) {
          addr <<= 2;
        }
        auto realIndex = op.memBase + addr;
        auto resultVal = memPool[realIndex];
        
        valuePool[op.result] = resultVal;
      }
    }
  } else {
    // unknow magic number
    assert(false);
  }
}

typedef struct {
  size_t netlistOffset;
  size_t netlistSize;
} SimMicroPartPtrs;

typedef struct {
  uint32_t numMParts;
  SimMicroPartPtrs *mPartInfo;
} SimMPartLevelInfo;

typedef struct {
  // Private data
  uint8_t *valuePool;
  uint16_t valuePoolSize;

  uint8_t *constVecPool;

  char* netlist;

  // Top level
  uint32_t numOpsL0RegRead;

  // Exec level
  uint32_t numExecLevels;
  SimMPartLevelInfo *execMPartLevelInfo;

  // Last level
  uint32_t numOpsLastRegWrite;
  uint32_t numOpsLastMemWrite;
  uint32_t numOpsLastPrint;
  uint32_t numOpsLastStop;

} SimPartitionPtrs;


__constant__ uint8_t lutContent[LUT_SIZE];

__device__ uint8_t *regPool, *memPool, *exchangePool;

__device__ bool shouldStop = false;
__device__ bool enablePrint = false;
__device__ uint32_t realCycles;

__device__ char **printMsgs;
__device__ SimPartitionPtrs *partitions;
__device__ uint32_t numParts;

uint8_t *regPool_device, *memPool_device;

std::vector<SimPartitionPtrs> gpuPartInfos;


extern __shared__ uint8_t sharedMem[];



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


  allocAndCopyVector(&regPool_device, design.regPool.data(), design.regPoolSize);
  allocAndCopyVector(&memPool_device, design.memPool.data(), design.memPoolSize);

  cudaMemcpyToSymbol(regPool, &regPool_device, sizeof(uint8_t*));
  cudaMemcpyToSymbol(memPool, &memPool_device, sizeof(uint8_t*));


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
    assert(eachPart.valuePoolSize <= UINT16_MAX);

    // copy const vec pool
    if (eachPart.constVecPool.empty()) {
      partInfo.constVecPool = nullptr;
    } else {
      allocAndCopyVector(&(partInfo.constVecPool), eachPart.constVecPool.data(), eachPart.constVecPool.size());
    }

    // copy operations

    // ops_l0
    partInfo.numOpsL0RegRead = eachPart.ops_l0_regRead.size();
    assert((partInfo.numOpsL0RegRead == 0));
    if (partInfo.numOpsL0RegRead != 0) {
      size_t memSize = eachPart.ops_l0_regRead.size() * sizeof(toucanGPUSim::CGRegReadMetaInfo);
      appendToNetlistVec(reinterpret_cast<const char*>(eachPart.ops_l0_regRead.data()), memSize);
    }


    // middle level mparts - serialize MicroParts
    std::vector<SimMPartLevelInfo> level_mpart_info;

    size_t numExecLevels = eachPart.exec_mParts.size();

    for (size_t level_id = 0; level_id < numExecLevels; level_id++) {
      const auto &mPartsInLevel = eachPart.exec_mParts[level_id];

      SimMPartLevelInfo levelInfo;
      levelInfo.numMParts = mPartsInLevel.size();

      // Create array of MicroPart pointers for this level
      std::vector<SimMicroPartPtrs> mPartPtrs;

      for (const auto &mPart : mPartsInLevel) {
        // Serialize each MicroPart and append to allNetlist for locality
        std::vector<char> mPartNetlist;
        copyMicroPartToNetlist(mPart, mPartNetlist);

        // Record the offset in allNetlist where this MicroPart starts
        size_t mPartOffset = allNetlist.size();

        // Append MicroPart netlist to the main netlist buffer
        appendToNetlistVec(mPartNetlist.data(), mPartNetlist.size());

        SimMicroPartPtrs mPartPtr;
        mPartPtr.netlistOffset = mPartOffset;
        mPartPtr.netlistSize = mPartNetlist.size();

        mPartPtrs.push_back(mPartPtr);
      }

      // Allocate GPU memory for the array of MicroPart pointers
      if (!mPartPtrs.empty()) {
        allocAndCopyVector(&(levelInfo.mPartInfo), mPartPtrs.data(), mPartPtrs.size() * sizeof(SimMicroPartPtrs));
      } else {
        levelInfo.mPartInfo = nullptr;
      }

      level_mpart_info.push_back(levelInfo);
    }

    partInfo.numExecLevels = numExecLevels;

    if (numExecLevels != 0) {
      size_t memSize = level_mpart_info.size() * sizeof(SimMPartLevelInfo);
      allocAndCopyVector(&(partInfo.execMPartLevelInfo), level_mpart_info.data(), memSize);
    } else {
      // no exec levels
      partInfo.execMPartLevelInfo = nullptr;
    }

    // last level - handle single regWrite and multiple other operations
    partInfo.numOpsLastRegWrite = 1; // Always 1 for op_last_regWrite
    partInfo.numOpsLastMemWrite = eachPart.ops_last_memWrite.size();
    partInfo.numOpsLastPrint = eachPart.ops_last_print.size();
    partInfo.numOpsLastStop = eachPart.ops_last_stop.size();

    // Serialize single regWrite operation
    size_t memSize = sizeof(toucanGPUSim::CGRegWriteMetaInfo);
    appendToNetlistVec(reinterpret_cast<const char*>(&eachPart.op_last_regWrite), memSize);

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
  assert(design.regionPartitionIds.size() == 1 && "For now only supports 1 region");
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
