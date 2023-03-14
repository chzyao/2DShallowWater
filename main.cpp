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
    ShallowWater test1;
    test1.SetParameters(argc,argv);



    return 0;
}
