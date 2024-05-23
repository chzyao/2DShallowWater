/**
 * @file main.cpp
 * @author Chris Ziyi Yao (chris.yao20@imperial.ac.uk)
 */

#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cblas.h>

#include "ShallowWater.h"
#include "Comm.h"

using namespace std;
namespace po = boost::program_options;


int main(int argc, char *argv[])
{
    // Allocate object on the heap
    ShallowWater *SWE = new ShallowWater;
    SWE -> SetParameters(argc, argv);
    SWE -> Solve(argc, argv);

    // Deallocation 
    delete SWE;
    
    return 0;
}
