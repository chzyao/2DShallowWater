#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <cblas.h>

#include "ShallowWater.h"

using namespace std;
namespace po = boost::program_options;


int main(int argc, char *argv[])
{
    // Allocate object on the heap
    ShallowWater *SWE = new ShallowWater;
    SWE -> SetParameters(argc, argv);
    SWE -> Solve();
    // cout << SWE->m_dt << endl;
    // cout << SWE->m_Nx << endl;
    // cout << SWE->m_ic << endl;

    delete SWE;
    return 0;
}
