#include "ToucanGPUGenDataTypes.h"
#include <random>
#include <array>

#include <iostream>

#include <cassert>


using namespace toucanGPUSim;

#define INIT_ARRAY_SIZE 1024



void SimDesignInfo::Init() {
  shouldStop = false;
  regPool.resize(regPoolSize, 0);
  // TODO: Consider faster init
  memPool.resize(memPoolSize, 0);
  exchangePool.resize(exchangePoolSize, 0);

  for (auto &eachPartition: parts) {
    // Expand value pool
    eachPartition.valuePool.resize(eachPartition.valuePoolSize);
  }
}

void SimDesignInfo::Randomize(uint32_t seed, SimDebugInfo &symbols) {
  // Randomize reg and mem
  std::random_device rd;
  std::mt19937 gen(seed);

  std::uniform_int_distribution<uint8_t> dist(0, 0xF);


  std::array<uint8_t, INIT_ARRAY_SIZE> randomNums;
  for (auto &elem: randomNums) {
    elem = dist(gen);
  }

  for (const auto &elem: symbols.regDebugInfo) {
    for (const auto &regInfo: elem.second) {
      auto regId = std::get<0>(regInfo);
      auto regWidth = std::get<1>(regInfo);
      auto mask = (1 << regWidth) - 1;

      regPool[regId] = randomNums[regId % INIT_ARRAY_SIZE] & mask;
    }
  }

  for (const auto &elem: symbols.memDebugInfo) {
    for (const auto &memInfo: elem.second) {
      auto memPos = std::get<0>(memInfo);
      auto memBitWidth = std::get<1>(memInfo);
      auto memDepth = std::get<2>(memInfo);
      auto mask = (1 << memBitWidth) - 1;
      for (size_t i = 0; i < memDepth; i++) {
        auto index = memPos + i;
        memPool[index] = randomNums[index % INIT_ARRAY_SIZE] & mask;
      }
    }
  }

}

