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

#include "ToucanGPUSim.h"




#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <getopt.h>


// using namespace std;


static uint64_t trace_count = 0;
bool verbose;
bool done_reset;




static void usage(const char * program_name) {
  printf("Usage: %s [EMULATOR OPTION]...\n",
         program_name);
  fputs("\
Dry run a toucan emitted binary.\n\
\n\
Mandatory arguments to long options are mandatory for short options too.\n\
\n\
EMULATOR OPTIONS\n\
  -h, --help               Display this help and exit\n\
  -m, --max-cycles=CYCLES  Kill the emulation after CYCLES\n\
       +max-cycles=CYCLES\n", stdout);
}


int main(int argc, char** argv) {
  unsigned random_seed = (unsigned)time(NULL) ^ (unsigned)getpid();
  uint64_t max_cycles = -1;
  int ret = 0;


  while (1) {
    static struct option long_options[] = {
      {"help",        no_argument,       0, 'h' },
      {"max-cycles",  required_argument, 0, 'm' }
    };
    int option_index = 0;
    int c = getopt_long(argc, argv, "-hm", long_options, &option_index);
    if (c == -1) break;
 retry:
    switch (c) {
      // Process long and short EMULATOR options
      case '?': usage(argv[0]);             return 1;
      case 'h': usage(argv[0]);             return 0;
      case 'm': max_cycles = atoll(optarg); break;
      // Realize that we've hit HTIF (HOST) arguments or error out
      default:
        c = '?';
        goto retry;
    }
  }


  random_seed = 0;


  auto sim = toucanGPUSim::ToucanSimulator();

  std::cout << "Loading design" << std::endl;

  sim.init(0, "GPUSimDesign.bin", "GPUSimIOSymbols.bin", false, false);

  std::cout << "Loading done" << std::endl;



  while (trace_count < max_cycles) {

    auto shouldStop = sim.eval();

    // if (shouldStop) {
    //   std::cout << "Simulation stop requested by design at cycle " << trace_count << std::endl;
    //   break;
    // }

    trace_count++;
  }

  return ret;

}

