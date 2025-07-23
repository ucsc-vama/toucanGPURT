#pragma once

#include <cstddef>
#include <fstream>
#include <vector>
#include "ToucanGPUGenDataTypes.h"
#include "GPUConsts.h"
#include "SimConfigs.h"

namespace toucanGPUSim {
  class ToucanSimulator {
    public:
    SimDesignInfo design;
    SimDebugInfo symbols;

    int init(const int gpu_id, const std::string designBinFilename, const std::string symbolBinFilename, bool shouldRandomize, bool enablePrint);

    void enableVCDDump(const std::string vcdFilename);

    bool eval();

    bool eval_free_running(uint32_t max_cycles);

    void setRegister(const std::string &signalName, uint64_t signalValue);

    uint64_t readRegister(const std::string &signalName);

    uint64_t readSignal(const std::string &signalName);

    void cleanup();

    #ifdef ENABLE_SIM_PROFILE
    void printProfile();
    #endif

    ~ToucanSimulator() {
      cleanup();
    }

    uint64_t cycle_cnt;

    private:
    int maxBlocksPerSMForSingleCycleKernel;
    int maxBlocksPerSMForMultiCycleKernel;
    int maxThreadsPerBlock;
    int numSMs;
    int maxNumPartsInEachRegion;

    int numBlocksForSingleCycleKernel;
    int numBlocksForMultiCycleKernel;

    int maxSharedMemoryPerSM;
    size_t maxValuePoolSize;
    size_t sharedMemPerBlock;


    bool dumpVCD;
    std::ofstream *vcdOfs;
    // ((part, valId, width), (part, valId, width), ..)
    std::vector<std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>> vcdSignalInfo;
    std::vector<uint32_t> vcdSignalBitWidth;
    std::vector<std::string> vcdSignalIdentifiers;
    std::vector<uint64_t> vcdSmallSignalCache;
    std::vector<std::vector<uint8_t>> vcdLargeSignalCache;

    int setupGPU(int gpu_id = 0);

    uint64_t getSmallSignalValue(const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>& info);

    std::vector<uint8_t> getLargeSignalValue(const std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>& info);

    void dumpVcdWorker(uint64_t cycle);
  };
}


