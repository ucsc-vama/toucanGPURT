#include "ToucanGPUGenDataTypes.h"
#include "SimEval.h"

#include <cstdint>
#include <fstream>
#include <iostream>

#include <ostream>
#include <cassert>


#include <iostream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>

#include "ToucanGPUSim.h"
#include "SimConfigs.h"



#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>


// using namespace std;



int main(int argc, char** argv) {
  int sim_cycles = 1000;

  if (argc < 2) {
    // std::cout << "Usage: " << argv[0] << " <integer>" << std::endl;
    // return 1;
  } else {
    sim_cycles = atoi(argv[1]);
  }

  auto sim = toucanGPUSim::ToucanSimulator();

  std::cout << "Loading design" << std::endl;

  sim.init(0, "GPUSimDesign.bin", "GPUSimIOSymbols.bin", false, false);

  std::cout << "Loading done" << std::endl;


  auto start = std::chrono::system_clock::now();

  auto ss = sim.eval_free_running(static_cast<uint32_t>(sim_cycles));
  // assert(ss == false);

  auto end = std::chrono::system_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  uint64_t time_ms = duration.count();
  std::cout << "Free running for " << sim_cycles << " cycles in " << time_ms << " ms\n";

  #ifdef ENABLE_SIM_PROFILE
  sim.printProfile();
  #endif

  return 0;

}

