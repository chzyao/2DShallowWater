/**
 * @file main.cpp
 * @author Chris (chris.yao20@imperial.ac.uk)
 * @version Initial version uploaded in March 2023. Last revised in May 2024.
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
