// #define NDEBUG

// Note: Enable async memcpy for reg write by uncomment this macro
// Seems not helpful for performance
// #define REG_WRITE_USE_ASYNC_MEMCPY

#include "ToucanGPUGenDataTypes.h"

#include "SimEval.h"
#include "SimConfigs.h"

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
#define LUT_SIZE 4642

#define MICROPART_REGULAR_MAX_LEVELS 160
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



typedef struct {
  uint32_t netlistOffset;
  uint32_t netlistSize;
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
  char* lastLevelNetlistPtr; // Direct pointer to last level operations

  // Top level
  uint32_t numOpsL0RegRead;
  uint32_t numOpsL0ExgRead;

  // Exec level
  uint32_t numExecLevels;
  SimMPartLevelInfo *execMPartLevelInfo;

  // Last level
  uint32_t numOpsLastExgWrite;
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

__device__ uint32_t numTotalParts;
__device__ uint32_t numParts_Region0, numParts_Region1;
__device__ uint32_t partTaskCounter;


#ifdef ENABLE_SIM_PROFILE
__device__ int64_t *profile_ticks_useful;
__device__ int64_t *profile_ticks_total;
int64_t *profile_ticks_useful_device;
int64_t *profile_ticks_total_device;
#endif

uint8_t *regPool_device, *memPool_device, *exchangePool_device;

std::vector<SimPartitionPtrs> gpuPartInfos;


extern __shared__ uint8_t sharedMem[];



__device__ __forceinline__ __int128 sign_extend_128(__int128 value, uint16_t  sign_bit_pos) {
  // Check if the sign bit is set
  bool is_negative = (value >> sign_bit_pos) & 1;
  if (!is_negative) return value;
  __int128 mask = ((__int128)1 << (sign_bit_pos + 1)) - 1;
  return value | ~mask;
}

/*
__device__ __forceinline__ uint8_t dummyLUT(uint16_t lutIndex, uint8_t op0, uint8_t op1, uint8_t op2) {
  switch (lutIndex) {
    case 0: {  // Pos for op nop: 0
      return op2;
    }
    case 546: {  // Pos for op and: 546
      return op1 & op2;
    }
    case 802: {  // Pos for op or: 802
      return op1 | op2;
    }
    case 1058: {  // Pos for op xor: 1058
      return op1 ^ op2;
    }
    case 16: {  // Pos for op rep1b: 16
      return (op2 == 0) ? 0 : 0xF;
    }
    case 34: {  // Pos for op add: 34
      return (op1 + op2) & 0xF;
    }
    case 2082: {  // Pos for op mux: 2082
      return (op0 == 1) ? op1 : op2;
    }
    case 2594: {  // Pos for op dshl: 2338
      return (((op1 << 4) | op2) >> (4 - op0)) & 0xF;
    }
    case 3618: {  // Pos for op dshr: 3362
      return (((op1 << 4) | op2) >> op0) & 0xF;
    }
    case 18: {  // Pos for op xorr: 18
      return __popc(op2);
    }
    case 2850: {  // Pos for op shl1: 2594
      return ((op1 << 1) | (op2 >> 3)) & 0xF;
    }
    case 3106: {  // Pos for op shl2: 2850
      return ((op1 << 2) | (op2 >> 2)) & 0xF;
    }
    case 3362: {  // Pos for op shl3: 3106
      return ((op1 << 3) | (op2 >> 1)) & 0xF;
    }
    case 1314: {  // Pos for op cmp_eq: 1314
      return (op1 == op2);
    }
    case 290: {  // Pos for op sub: 290
      return (op1 - op2) & 0xF;
    }
    case 1570: {  // Pos for op cmp_ult: 1570
      return (op1 < op2);
    }
    case 1826: {  // Pos for op cmp_slt4b: 1826
      int8_t op1s = op1;
      int8_t op2s = op2;
      if ((op1s & 0b1000) != 0) {
        op1s |= 0xF0;
      }
      if ((op2s & 0b1000) != 0) {
        op2s |= 0xF0;
      }
      return op1s < op2s;
    }
    default: {
      printf("Unknown lut pos: %d\n", static_cast<int>(lutIndex));
      assert(false);
    }
  }
  return 0;
}
*/


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
      
      uint8_t resultVal = 0;
      if (lutIndex == LUT_NOP_INDEX) {
        resultVal = op2Val;
      } else {
        uint16_t lutPos = lutIndex+ ((static_cast<uint16_t>(op0Val) << 8) | (op1Val << 4) | op2Val);
        resultVal = lutContent[lutPos];
        // resultVal = dummyLUT(lutIndex, op0Val, op1Val, op2Val);
      }

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
        uint32_t opRaw = op.getPacked();
        op2 = opRaw & 0x3F;
        op1 = (opRaw >> 6) & 0x3F;
        op2 = (opRaw >> 12) & 0x3F;
        lutIndex = opRaw >> 18;
      }

      // Every thread in the wrap should participate in shuffle
      // T __shfl_sync(unsigned mask, T var, int srcLane, int width=warpSize);
      auto op0Val = __shfl_sync(0xFFFFFFFF, thisLaneShuffleVal, op0);
      auto op1Val = __shfl_sync(0xFFFFFFFF, thisLaneShuffleVal, op1);
      auto op2Val = __shfl_sync(0xFFFFFFFF, thisLaneShuffleVal, op2);

      // update value if it's actually a const
      if (op0 < 32) {
        // assert(op0 < 16);
        op0Val = op0;
      }
      if (op1 < 32) {
        // assert(op1 < 16);
        op1Val = op1;
      }
      if (op2 < 32) {
        // assert(op2 < 16);
        op2Val = op2;
      }

      uint8_t resultVal = 0;
      if (lutIndex == LUT_NOP_INDEX) {
        resultVal = op2Val;
      } else {
        uint16_t lutPos = lutIndex + ((static_cast<uint16_t>(op0Val) << 8) | (op1Val << 4) | op2Val);
        resultVal = lutContent[lutPos];
        // resultVal = dummyLUT(lutIndex, op0Val, op1Val, op2Val);
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
      // assert(shuffleId < 16);
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
      
      auto index0Val = static_cast<uint32_t>((op.index0 < 16) ? op.index0 : valuePool[op.index0]);
      auto index1Val = static_cast<uint32_t>((op.index1 < 16) ? op.index1 : valuePool[op.index1]);
      auto index2Val = static_cast<uint32_t>((op.index2 < 16) ? op.index2 : valuePool[op.index2]);
      auto index3Val = static_cast<uint32_t>((op.index3 < 16) ? op.index3 : valuePool[op.index3]);
      
      
      
      uint32_t vecOffset = ((index0Val << 12) | (index1Val << 8) | (index2Val << 4) | index3Val) + op.offset;
      
      uint8_t resultVal = 0;

      if (vecOffset < op.vecLength) {
        if (op.isConstVec) {
          resultVal = constVecPool[op.vecBase + vecOffset];
        } else {
          resultVal = valuePool[op.vecBase + vecOffset];
        }
      } else {
        auto outRangeVal = (op.outRangeValue < 16) ? op.outRangeValue : valuePool[op.outRangeValue];
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
      // assert(vecLength * 4 <= 128);

      bool isV1Const = (op.isV1V2Const & 0b10) != 0;
      bool isV2Const = (op.isV1V2Const & 0b01) != 0;

      auto v1Base = op.vec1Base;
      auto v2Base = op.vec2Base;

      __int128 v1Val = 0;
      __int128 v2Val = 0;

      for (int i = 0; i < vecLength; i++) {
        __int128 temp;

        uint8_t v1_seg = isV1Const ? constVecPool[v1Base + i] : valuePool[v1Base + i];
        uint8_t v2_seg = isV2Const ? constVecPool[v2Base + i] : valuePool[v2Base + i];

        temp = v1_seg;
        v1Val = v1Val | (temp << (i * vecElemWidth));

        temp = v2_seg;
        v2Val = v2Val | (temp << (i * vecElemWidth));
      }

      v1Val = sign_extend_128(v1Val, vecLength * vecElemWidth - 1);
      v2Val = sign_extend_128(v2Val, vecLength * vecElemWidth - 1);

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
      
      auto enVal = (op.en < 16) ? op.en : valuePool[op.en];
      if (enVal != 0) {
        uint32_t addr = 0;
        for (uint32_t j = 0; j < 8; j++) {
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

__device__ void evalPartL0RegRead(
  uint8_t * __restrict valuePool, 
  const toucanGPUSim::CGRegReadMetaInfo * __restrict topLevelRegReadOps,
  const uint32_t numRegReads) {

  auto block = cg::this_thread_block();
  auto thread_rank = block.thread_rank();
  auto threads_in_block = block.size();

  // Eval reg reads
  for (uint32_t op_pos = thread_rank; op_pos < numRegReads; op_pos += threads_in_block) {
    // reg read
    const auto &op = topLevelRegReadOps[op_pos];
    auto regValId = op.reg;
    auto resultId = op.result;

    auto regVal = regPool[regValId];
    valuePool[resultId] = regVal;
  }
}

__device__ void evalPartL0ExgRead(
  uint8_t * __restrict valuePool, 
  const toucanGPUSim::CGExchangeReadMetaInfo * __restrict topLevelExgReadOps,
  const uint32_t numExgReads) {

  auto block = cg::this_thread_block();
  auto thread_rank = block.thread_rank();
  auto threads_in_block = block.size();

  // Eval exchange reads
  for (uint32_t op_pos = thread_rank; op_pos < numExgReads; op_pos += threads_in_block) {
    // reg read
    const auto &op = topLevelExgReadOps[op_pos];
    auto exgValId = op.exchange;
    auto resultId = op.result;

    auto exgVal = exchangePool[exgValId];
    valuePool[resultId] = exgVal;
  }
}


__device__ void evalLastLevel(
  uint8_t * __restrict valuePool,
  const char * __restrict lastLevelNetlistPtr,
  const uint32_t numRegWriteOps,
  const uint32_t numMemWriteOps,
  const uint32_t numPrintOps,
  const uint32_t numStopOps) {

  auto block = cg::this_thread_block();
  auto thread_rank = block.thread_rank();
  auto threads_in_block = block.size();

  // Calculate netlist locations internally
  auto netlist_regWrite = align_pointer(const_cast<char*>(lastLevelNetlistPtr));
  auto netlist_memWrite = align_pointer(netlist_regWrite + (numRegWriteOps * sizeof(toucanGPUSim::CGRegWriteMetaInfo)));
  auto netlist_print = align_pointer(netlist_memWrite + (numMemWriteOps * sizeof(toucanGPUSim::CGMemWriteMetaInfo)));
  auto netlist_stop = align_pointer(netlist_print + (numPrintOps * sizeof(toucanGPUSim::CGPrintMetaInfo)));

  const auto *regWriteOps = reinterpret_cast<const toucanGPUSim::CGRegWriteMetaInfo*>(netlist_regWrite);
  const auto *memWriteOps = reinterpret_cast<const toucanGPUSim::CGMemWriteMetaInfo*>(netlist_memWrite);
  const auto *printOps = reinterpret_cast<const toucanGPUSim::CGPrintMetaInfo*>(netlist_print);
  const auto *stopOps = reinterpret_cast<const toucanGPUSim::CGStopMetaInfo*>(netlist_stop);

  // assert(numRegWriteOps <= 1 && "Only support single reg write in last level");
  if (numRegWriteOps != 0){
    const auto &op = regWriteOps[0];

#ifdef REG_WRITE_USE_ASYNC_MEMCPY
    uint32_t bytesToCopy = (op.count + 15) & 0xFFFFFFF0; // Fix: should be +15 for proper 16-byte alignment

    cg::memcpy_async(block, regPool + op.reg, valuePool + op.dat, bytesToCopy); // Fix: correct src/dst order
#else
    uint32_t intsToCopy = (op.count + 3) / 4;
    for (uint32_t i = thread_rank; i < intsToCopy; i += threads_in_block) {
      uint32_t poolOffset = (op.dat >> 2) + i;
      auto val = reinterpret_cast<uint32_t*>(valuePool)[poolOffset];

      uint32_t regOffset = (op.reg >> 2) + i;
      reinterpret_cast<uint32_t*>(regPool)[regOffset] = val;
    }
#endif
  }


  for (uint32_t op_pos = thread_rank; op_pos < numMemWriteOps; op_pos += threads_in_block) {
    const auto op = memWriteOps[op_pos];

    // memwrite
    auto enVal = (op.en < 16) ? op.en : valuePool[op.en];
    if (enVal != 0) {
      auto datVal = valuePool[op.dat];
      auto addrVecId = op.addrVec;

      uint32_t addr = 0;
      for (uint32_t i = 0; i < 8; i++) {
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


  for (uint32_t op_pos = thread_rank; op_pos < numPrintOps; op_pos += threads_in_block) {
    if (!enablePrint) continue;
    const auto op = printOps[op_pos];
    auto enVal = (op.en < 16) ? op.en : valuePool[op.en];
    if (enVal != 0) {
      auto msgPtr = printMsgs[op.msg];
      printf(msgPtr);
    }
  }

  for (uint32_t op_pos = thread_rank; op_pos < numStopOps; op_pos += threads_in_block) {
    const auto op = stopOps[op_pos];

    // stop
    auto enVal = (op.en < 16) ? op.en : valuePool[op.en];
    if (enVal != 0) {
      shouldStop = true;
    }
  }

#ifdef REG_WRITE_USE_ASYNC_MEMCPY
  __threadfence();
  cg::wait(block); // Wait for all copies to complete
#endif

}


__device__ void evalLastLevelExgWrite(
  uint8_t * __restrict valuePool,
  const char * __restrict lastLevelNetlistPtr) {

  auto block = cg::this_thread_block();
  auto thread_rank = block.thread_rank();
  auto threads_in_block = block.size();

  // Calculate netlist locations internally
  auto netlist_exgWrite = align_pointer(const_cast<char*>(lastLevelNetlistPtr));


  const auto *exgWriteOps = reinterpret_cast<const toucanGPUSim::CGExchangeWriteMetaInfo*>(netlist_exgWrite);


  // assert(numExgWriteOps == 1 && "Only support single reg write in last level");
  {
    const auto &op = exgWriteOps[0];

    uint32_t intsToCopy = (op.count + 3) / 4;
    for (uint32_t i = thread_rank; i < intsToCopy; i += threads_in_block) {
      uint32_t poolOffset = (op.dat >> 2) + i;
      auto val = reinterpret_cast<uint32_t*>(valuePool)[poolOffset];

      uint32_t exgOffset = (op.exchange >> 2) + i;
      reinterpret_cast<uint32_t*>(exchangePool)[exgOffset] = val;
    }
  }

}


// Note: launch in 1D
__device__ void evalEachPartition(uint32_t partId) {
  auto &partPtrs = partitions[partId];

  auto block = cg::this_thread_block();
  auto thread_rank = block.thread_rank();
  auto threads_in_block = block.size();

  uint8_t *localValuePool = reinterpret_cast<uint8_t*>(sharedMem);
  __shared__ uint32_t warpTaskCounter;
  
  // Note: Not necessary. Temporal values are written first then read, initialize unneeded
  // Note: However, save intermediate values may be needed if want to dump waveform.

  // // Copy constants from valuePool to shared memory
  // for (size_t data_pos = thread_rank; data_pos < partPtrs.valuePoolSize; data_pos += threads_in_block) {
  //   localValuePool[data_pos] = partPtrs.valuePool[data_pos];
  // }
  // __syncthreads();

  auto netlist_ptr = partPtrs.netlist;
  char* netlist_current_pos = netlist_ptr;

  // 1. Evaluate L0 (register reads)
  if (partPtrs.numOpsL0RegRead > 0) {
    auto netlist_regRead = align_pointer(netlist_current_pos);
    evalPartL0RegRead(localValuePool, 
               reinterpret_cast<const toucanGPUSim::CGRegReadMetaInfo*>(netlist_regRead), 
               partPtrs.numOpsL0RegRead);
    netlist_current_pos = netlist_regRead + (partPtrs.numOpsL0RegRead * sizeof(toucanGPUSim::CGRegReadMetaInfo));
    __syncthreads();
  }
  if (partPtrs.numOpsL0ExgRead > 0) {
    auto netlist_exgRead = align_pointer(netlist_current_pos);
    evalPartL0ExgRead(localValuePool, 
               reinterpret_cast<const toucanGPUSim::CGExchangeReadMetaInfo*>(netlist_exgRead), 
               partPtrs.numOpsL0ExgRead);
    netlist_current_pos = netlist_exgRead + (partPtrs.numOpsL0ExgRead * sizeof(toucanGPUSim::CGExchangeReadMetaInfo));
    __syncthreads();
  }

  // 2. Evaluate execution levels using MicroParts
  for (uint32_t exec_level_id = 0; exec_level_id < partPtrs.numExecLevels; exec_level_id++) {
    const auto &levelInfo = partPtrs.execMPartLevelInfo[exec_level_id];
    
    // Each warp (32 threads) processes one MicroPart
    // uint32_t warp_id = thread_rank / 32;
    // uint32_t warps_per_block = (threads_in_block + 31) / 32;
    uint32_t num_mparts_this_level = levelInfo.numMParts;
    auto lane_id = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block()).thread_rank();
    
    // Distribute MicroParts across warps with proper work distribution
    // Each warp dynamically pick next task

    // reset task counter
    if (thread_rank == 0) {
      warpTaskCounter = 0;
    }
    __syncthreads();

    while (true) {
      uint32_t mpart_id;

      // fetch next mpart id and broadcast to all threads in this warp
      if (lane_id == 0) {
        // fetch tasks
        mpart_id = atomicAdd(&warpTaskCounter, 1);
      }
      mpart_id = __shfl_sync(0xFFFFFFFF, mpart_id, 0);

      // Complete
      if (mpart_id >= num_mparts_this_level) break;

      const auto &mPartPtr = levelInfo.mPartInfo[mpart_id];
      // Calculate actual netlist pointer from base + offset
      char *mPartNetlistPtr = partPtrs.netlist + mPartPtr.netlistOffset;
      // This warp processes this MicroPart
      evalSingleMicroPart(localValuePool, partPtrs.constVecPool, mPartNetlistPtr);
    }
    __syncthreads();
  }

  // 3. Evaluate last level operations using direct pointer
  if (partPtrs.numOpsLastExgWrite != 0) {
    // exchange write
    evalLastLevelExgWrite(localValuePool, partPtrs.lastLevelNetlistPtr);
  } else {
    evalLastLevel(
      localValuePool, 
      partPtrs.lastLevelNetlistPtr,
      partPtrs.numOpsLastRegWrite, 
      partPtrs.numOpsLastMemWrite, 
      partPtrs.numOpsLastPrint, 
      partPtrs.numOpsLastStop);
  }

}



__global__ void evalSingleCycle() {

  uint32_t block_rank = blockIdx.x;
  uint32_t blocks_in_grid = gridDim.x;

  uint32_t partId = block_rank;
  for (; partId < numParts_Region0; partId += blocks_in_grid) {
    if (partId < numParts_Region0) {
      evalEachPartition(partId);
    }
  }
  cooperative_groups::this_grid().sync();

  for (partId = block_rank + numParts_Region0; partId < numTotalParts; partId += blocks_in_grid) {
    if (partId < numTotalParts) {
      evalEachPartition(partId);
    }
  }
  cooperative_groups::this_grid().sync();
}

__global__ void evalFreeRunningNCycles(uint32_t cycleCnt) {
  for (uint32_t cycle = 0; cycle < cycleCnt; cycle++) {
    uint32_t block_rank = blockIdx.x;
    uint32_t blocks_in_grid = gridDim.x;

    #ifdef ENABLE_SIM_PROFILE
    int64_t start_clock = clock64();
    #endif

    uint32_t partId = block_rank;

    for (; partId < numParts_Region0; partId += blocks_in_grid) {
      if (partId < numParts_Region0) {
        evalEachPartition(partId);

        #ifdef ENABLE_SIM_PROFILE
        if (cycle >= PROFILE_START_CYCLE && cycle < (PROFILE_START_CYCLE + PROFILE_COLLECT_CYCLE)) {
          if (cg::this_thread_block().thread_rank() == 0) {
            int64_t end_clock = clock64();
            int64_t useful_cycles = end_clock - start_clock;

            uint32_t profile_cycle = cycle - PROFILE_START_CYCLE;

            int64_t average_useful_cycle = ((profile_ticks_useful[partId] * profile_cycle) + useful_cycles) / (profile_cycle + 1);
            profile_ticks_useful[partId] = average_useful_cycle;
          }
        }
        #endif
      }
    }


    cooperative_groups::this_grid().sync();

    #ifdef ENABLE_SIM_PROFILE

    for (; partId < numParts_Region0; partId += blocks_in_grid) {
      if (partId < numParts_Region0) {

        if (cycle >= PROFILE_START_CYCLE && cycle < (PROFILE_START_CYCLE + PROFILE_COLLECT_CYCLE)) {
          if (cg::this_thread_block().thread_rank() == 0) {
            int64_t end_clock = clock64();
            int64_t total_cycles = end_clock - start_clock;

            uint32_t profile_cycle = cycle - PROFILE_START_CYCLE;

            int64_t average_total_cycle = ((profile_ticks_total[partId] * profile_cycle) + total_cycles) / (profile_cycle + 1);
            profile_ticks_total[partId] = average_total_cycle;
          }
        }
      }
    }

    start_clock = clock64();
    #endif

    for (partId = block_rank + numParts_Region0; partId < numTotalParts; partId += blocks_in_grid) {
      if (partId < numTotalParts) {
        evalEachPartition(partId);

        #ifdef ENABLE_SIM_PROFILE
        if (cycle >= PROFILE_START_CYCLE && cycle < (PROFILE_START_CYCLE + PROFILE_COLLECT_CYCLE)) {
          if (cg::this_thread_block().thread_rank() == 0) {
            int64_t end_clock = clock64();
            int64_t useful_cycles = end_clock - start_clock;

            uint32_t profile_cycle = cycle - PROFILE_START_CYCLE;

            int64_t average_useful_cycle = ((profile_ticks_useful[partId] * profile_cycle) + useful_cycles) / (profile_cycle + 1);
            profile_ticks_useful[partId] = average_useful_cycle;
          }
        }
        #endif
      }
    }

    cooperative_groups::this_grid().sync();

    #ifdef ENABLE_SIM_PROFILE
    for (partId = block_rank + numParts_Region0; partId < numTotalParts; partId += blocks_in_grid) {
      if (partId < numTotalParts) {
        if (cycle >= PROFILE_START_CYCLE && cycle < (PROFILE_START_CYCLE + PROFILE_COLLECT_CYCLE)) {
          if (cg::this_thread_block().thread_rank() == 0) {
            int64_t end_clock = clock64();
            int64_t total_cycles = end_clock - start_clock;

            uint32_t profile_cycle = cycle - PROFILE_START_CYCLE;

            int64_t average_total_cycle = ((profile_ticks_total[partId] * profile_cycle) + total_cycles) / (profile_cycle + 1);
            profile_ticks_total[partId] = average_total_cycle;
          }
        }
      }
    }
    #endif

    // update cycle counter
    auto thread_rank = cooperative_groups::this_grid().thread_rank();
    if (thread_rank == 0) {
      realCycles += 1;
    }
    // if (shouldStop) {
    //   return;
    // }
  }

}

// Each thread block find next job using atomic
__global__ void evalFreeRunningNCycles_Large(uint32_t cycleCnt) {
  for (uint32_t cycle = 0; cycle < cycleCnt; cycle++) {

    __shared__ uint32_t part_id;

    // #ifdef ENABLE_SIM_PROFILE
    // uint32_t block_rank = blockIdx.x;
    // int64_t start_clock = clock64();
    // #endif

    if (cg::this_grid().thread_rank() == 0) {
      partTaskCounter = 0;
    }
    cg::this_grid().sync();

    while (true) {
      if (cg::this_thread_block().thread_rank() == 0) {
        uint32_t part_id_local = atomicAdd(&partTaskCounter, 1);
        part_id = part_id_local;
      }

      __syncthreads();

      if (part_id >= numParts_Region0) break;

      evalEachPartition(part_id);
    }

    cooperative_groups::this_grid().sync();

    if (cg::this_grid().thread_rank() == 0) {
      partTaskCounter = numParts_Region0;
    }
    cg::this_grid().sync();

    while (true) {
      if (cg::this_thread_block().thread_rank() == 0) {
        uint32_t part_id_local = atomicAdd(&partTaskCounter, 1);
        part_id = part_id_local;
      }

      __syncthreads();

      if (part_id >= numTotalParts) break;

      evalEachPartition(part_id);
    }

    // update cycle counter
    auto thread_rank = cooperative_groups::this_grid().thread_rank();
    if (thread_rank == 0) {
      realCycles += 1;
    }
    // if (shouldStop) {
    //   return;
    // }
  }

}

uint64_t read_reg_from_gpu(const std::vector<std::tuple<uint32_t, uint32_t>>& signalLocs) {
  uint64_t result = 0;

  for(auto it = signalLocs.begin(); it != signalLocs.end(); ++it) {
    auto pos = std::get<0>(*it);
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


#ifdef ENABLE_SIM_PROFILE
void copy_profile_data(int64_t *usefulTicks, int64_t *totalTicks, size_t count) {
  gpuErrchk(cudaMemcpy(usefulTicks, profile_ticks_useful_device, count * sizeof(int64_t), cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(totalTicks, profile_ticks_total_device, count * sizeof(int64_t), cudaMemcpyDeviceToHost));
}

#endif

template <typename T>
static void allocAndCopyVector(T **devicePtr, const void *data, const size_t size) {
  cudaMalloc(devicePtr, size);
  cudaMemcpy(*devicePtr, data, size, cudaMemcpyHostToDevice);
}

void copy_netlist_to_gpu(toucanGPUSim::SimDesignInfo &design) {
  // reset cycle count
  uint32_t realCycles_host = 0;
  cudaMemcpyToSymbol(realCycles, &realCycles_host, sizeof(uint32_t));
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

  std::vector<uint8_t> exgPool_local;
  exgPool_local.resize(design.exchangePoolSize, 0);
  allocAndCopyVector(&exchangePool_device, exgPool_local.data(), design.exchangePoolSize);
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
    if (partInfo.numOpsL0RegRead != 0) {
      size_t memSize = eachPart.ops_l0_regRead.size() * sizeof(toucanGPUSim::CGRegReadMetaInfo);
      appendToNetlistVec(reinterpret_cast<const char*>(eachPart.ops_l0_regRead.data()), memSize);
    }
    // exg read
    partInfo.numOpsL0ExgRead = eachPart.ops_l0_exchangeRead.size();
    if (partInfo.numOpsL0ExgRead != 0) {
      size_t memSize = eachPart.ops_l0_exchangeRead.size() * sizeof(toucanGPUSim::CGExchangeReadMetaInfo);
      appendToNetlistVec(reinterpret_cast<const char*>(eachPart.ops_l0_exchangeRead.data()), memSize);
    }
    assert((partInfo.numOpsL0ExgRead == 0) != (partInfo.numOpsL0RegRead == 0));


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
    partInfo.numOpsLastRegWrite = eachPart.op_last_regWrite.count != 0;
    partInfo.numOpsLastExgWrite = eachPart.op_last_exchangeWrite.count != 0;
    partInfo.numOpsLastMemWrite = eachPart.ops_last_memWrite.size();
    partInfo.numOpsLastPrint = eachPart.ops_last_print.size();
    partInfo.numOpsLastStop = eachPart.ops_last_stop.size();

    // Record where last level operations start in the netlist
    size_t lastLevelOffset = allNetlist.size();

    // Serialize single regWrite operation
    if (partInfo.numOpsLastRegWrite != 0) {
      size_t memSize = sizeof(toucanGPUSim::CGRegWriteMetaInfo);
      appendToNetlistVec(reinterpret_cast<const char*>(&eachPart.op_last_regWrite), memSize);
    }

    if (partInfo.numOpsLastExgWrite != 0) {
      size_t memSize = sizeof(toucanGPUSim::CGExchangeWriteMetaInfo);
      appendToNetlistVec(reinterpret_cast<const char*>(&eachPart.op_last_exchangeWrite), memSize);
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

    // Set the lastLevelNetlistPtr to point directly to last level operations
    partInfo.lastLevelNetlistPtr = partInfo.netlist + lastLevelOffset;

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
  if (design.regionPartitionIds.size() != 2) {
    std::cout << "For now only supports 2 region, " << design.regionPartitionIds.size() << " region in design file\n";
  }
  assert(design.regionPartitionIds.size() == 2 && "For now only supports 2 region");
  std::vector<uint32_t> numParts;
  size_t _partId = 0;
  for (const auto &eachRegionParts: design.regionPartitionIds) {
    for (const auto partId: eachRegionParts) {
      assert(partId == _partId);
      _partId++;
    }
    numParts.push_back(eachRegionParts.size());
  }
  // Should have exact 1 region
  assert(numParts.size() == 2);
  cudaMemcpyToSymbol(numParts_Region0, &numParts[0], sizeof(uint32_t));
  cudaMemcpyToSymbol(numParts_Region1, &numParts[1], sizeof(uint32_t));
  uint32_t totalParts = design.parts.size();
  assert(totalParts = numParts[0] + numParts[1]);
  cudaMemcpyToSymbol(numTotalParts, &totalParts, sizeof(uint32_t));

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

#ifdef ENABLE_SIM_PROFILE
  {
    std::vector<uint64_t> zeroTicks;
    zeroTicks.resize(totalParts, 0);

    cudaMalloc(&profile_ticks_useful_device, sizeof(int64_t) * totalParts);
    cudaMemcpy(profile_ticks_useful_device, zeroTicks.data(), sizeof(int64_t) * totalParts, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(profile_ticks_useful, &profile_ticks_useful_device, sizeof(int64_t*));
    cudaMalloc(&profile_ticks_total_device, sizeof(int64_t) * totalParts);
    cudaMemcpy(profile_ticks_total_device, zeroTicks.data(), sizeof(int64_t) * totalParts, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(profile_ticks_total, &profile_ticks_total_device, sizeof(int64_t*));
  }
#endif

  gpuErrchk(cudaDeviceSynchronize());
}
