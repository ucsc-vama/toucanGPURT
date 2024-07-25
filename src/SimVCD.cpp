#include "ToucanGPUSim.h"
#include <cstddef>
#include <fstream>

#include <tuple>
#include <iostream>

#include <cassert>
#include <unordered_map>


using namespace toucanGPUSim;

#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <set>

struct VcdModule {
  std::map<std::string, VcdModule> subModules;
  std::vector<std::string> signals;
  std::string moduleName;
};

// Parse the flattened signal names and build a module hierarchy
static void parseSignalNames(const std::vector<std::string>& signalNames, VcdModule& root) {
  for (const auto& fullname : signalNames) {
    std::istringstream iss(fullname);
    std::string part;
    std::string nextPart;
    VcdModule* current = &root;

    while (getline(iss, part, '.')) {
      auto remainingStr = iss.str();
      if (remainingStr.find('.') != std::string::npos) {
        // Not the last part, so it's a module name
        if (!current->subModules.contains(part)) {
          // A new module
          current->subModules[part] = {};
        }
        current = &current->subModules[part];

        auto nextStr = iss.str().substr(part.length() + 1);
        iss = std::istringstream(nextStr);
      } else {
        // Last part, so it's a signal name
        current->signals.push_back(part);
      }
    }
  }
}

// Helper function to write the module hierarchy to a VCD file
static void writeModule(std::ofstream* vcdFile, const std::string& moduleName, const VcdModule& module, const std::string& modulePrefix, const std::unordered_map<std::string, std::tuple<uint32_t, std::string>>& signalMetaInfos) {
  if (!moduleName.empty()) {
    *vcdFile << "$scope module " << moduleName << " $end\n";
    for (const auto& signalName : module.signals) {
      std::string signalFullName = modulePrefix + "." + signalName;

      auto sigMeta = signalMetaInfos.at(signalFullName);
      auto sigWidth = std::get<0>(sigMeta);
      auto sigIdentifier = std::get<1>(sigMeta);

      *vcdFile << "$var wire " << sigWidth << " " << sigIdentifier << " " << signalName << " $end\n";
    }
  }

  for (const auto& sub : module.subModules) {
    auto nextModuleName = sub.first;
    std::string nextPrefix = modulePrefix;
    if (!moduleName.empty()) {
      nextPrefix += ".";
    }
    nextPrefix += nextModuleName;
    writeModule(vcdFile, nextModuleName, sub.second, nextPrefix, signalMetaInfos);
  }
  if (!moduleName.empty()) {
    *vcdFile << "$upscope $end\n";
  }
  
}



static std::string getNextIdentifier(uint32_t& id) {
  std::string identifier;
  uint32_t currentId = id;
  do {
    // Starting from '!' to avoid non-printable characters
    identifier = char('!' + currentId % 94) + identifier; 
    currentId /= 94;
  } while (currentId != 0);
  return identifier;
}

void ToucanSimulator::enableVCDDump(const std::string vcdFilename) {
  dumpVCD = true;
  vcdOfs = new std::ofstream(vcdFilename, std::ios::out);


  // Sort for performance
  std::vector<std::tuple<std::string, std::vector<std::tuple<uint32_t, uint32_t, uint32_t>>>> sortedSignalDebugInfo(symbols.signalDebugInfo.begin(), symbols.signalDebugInfo.end());

  std::sort(sortedSignalDebugInfo.begin(), sortedSignalDebugInfo.end(), [](const auto& a, const auto& b) {
      auto a_first_info = std::get<1>(a)[0];
      auto b_first_info = std::get<1>(b)[0];

      auto a_partId = static_cast<uint64_t>(std::get<0>(a_first_info));
      auto a_valId = std::get<1>(a_first_info);
      auto b_partId = static_cast<uint64_t>(std::get<0>(b_first_info));
      auto b_valId = std::get<1>(b_first_info);
      return ((a_partId << 32) + a_valId) < ((b_partId << 32) + b_valId);
    });
  
  auto signalCount = sortedSignalDebugInfo.size();

  //
  vcdSignalInfo.reserve(signalCount);
  for (auto eachSignal: sortedSignalDebugInfo) {
    vcdSignalInfo.push_back(std::get<1>(eachSignal));
  }
  
  // identifiers & bitWidth

  vcdSignalIdentifiers.reserve(signalCount);
  vcdSignalBitWidth.reserve(signalCount);

  for (uint32_t i = 0; i < signalCount; i++) {
    vcdSignalIdentifiers.push_back(getNextIdentifier(i));

    auto &sigInfo = vcdSignalInfo[i];
    uint32_t sigWidth = 0;
    for (auto &ei: sigInfo) {
      sigWidth += std::get<2>(ei);
    }
    vcdSignalBitWidth.push_back(sigWidth);
  }

  // clear vcd signal cache
  vcdSmallSignalCache.resize(signalCount, 0);
  vcdLargeSignalCache.resize(signalCount);


  *vcdOfs << "$date\n    " << __DATE__ << " " << __TIME__ << "\n$end\n"
          << "$version\n    VCD generator\n$end\n"
          << "$timescale\n    1ps\n$end\n"; // Adjust timescale as needed
          // << "$scope module logic $end\n";




  std::vector<std::string> signalNames;
  std::unordered_map<std::string, std::tuple<uint32_t, std::string>> signalMetaInfos;


  for (size_t i = 0; i < signalCount; ++i) {
    auto sigName = std::get<0>(sortedSignalDebugInfo[i]);
    // auto sigInfo = std::get<1>(sortedSignalDebugInfo[i]);
    uint32_t sigWidth = vcdSignalBitWidth[i];
    auto sigIdentifier = vcdSignalIdentifiers[i];
    
    signalNames.push_back(sigName);
    // auto sigInfo = std::make_tuple(sigWidth, sigIdentifier);
    signalMetaInfos[sigName] = {sigWidth, sigIdentifier};
  }

  VcdModule root;
  parseSignalNames(signalNames, root);

  for (const auto& module : root.subModules) {
    writeModule(vcdOfs, module.first, module.second, module.first, signalMetaInfos);
  }

  *vcdOfs << "$enddefinitions $end\n"
          << "$dumpvars\n";

}

static char buffer[65];
static char *toBinaryString(uint64_t dat, uint32_t bitWidth) {
  assert(bitWidth <= 64);
  buffer[bitWidth] = 0;
  for (int cursor = bitWidth - 1; cursor >= 0; cursor--) {
    buffer[cursor] = '0' + (dat & 1);
    dat = dat >> 1;
  }
  // assert(dat == 0);
  return buffer;
}


void ToucanSimulator::dumpVcdWorker(uint64_t cycle) {
  *vcdOfs << "#" << cycle << "\n"; 
  for (uint32_t i = 0; i < vcdSignalIdentifiers.size(); i++) {
    // for each signal
    auto &identifier = vcdSignalIdentifiers[i];
    auto &locInfo = vcdSignalInfo[i];
    auto bitWidth = vcdSignalBitWidth[i];

    if (bitWidth <= 64) {
      // small signal
      auto signalValue = getSmallSignalValue(locInfo);
      auto lastSignalValue = vcdSmallSignalCache[i];
      bool valueChanged = (cycle == 0) ? true : (signalValue != lastSignalValue);

      if (valueChanged) {
        vcdSmallSignalCache[i] = signalValue;
        if (bitWidth == 1) {
          *vcdOfs << signalValue << identifier << "\n";
        } else {
          *vcdOfs << "b" << toBinaryString(signalValue, bitWidth) << " " << identifier << "\n";
        }
      }
    } else {
      // bitWidth > 64.
      auto signalValues = getLargeSignalValue(locInfo);
      auto lastSignalValue = vcdLargeSignalCache[i];
      bool valueChanged = (cycle == 0) ? true : lastSignalValue != signalValues;

      if (valueChanged) {
        vcdLargeSignalCache[i] = std::move(signalValues);
        
        *vcdOfs << "b";

        for (auto &ei: locInfo) {
          auto partId = std::get<0>(ei);
          auto valId = std::get<1>(ei);
          auto sectionWidth = std::get<2>(ei);
          auto sectionVal = design.parts[partId].valuePool[valId];
          assert(sectionVal <= 0xF);
          *vcdOfs << toBinaryString(sectionVal, sectionWidth);
        }

        *vcdOfs << " " << identifier << "\n";
      }
    }
  }
}

void ToucanSimulator::cleanup() {
  if (vcdOfs != nullptr) {
    vcdOfs->close();
    vcdOfs = nullptr;
  }
}
