#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <iomanip>
#include <fstream>

#include "ShallowWater.h"

using namespace std;
namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    cout << "Goodbye World" << endl;
    // Read parameters from command line
    po::options_description options("Available Options.");
    options.add_options()
        ("help","Display help message")
        ("dt", po::value<double>()->default_value(0.0),"Time-step to use")
        ("T", po::value<double>()->default_value(0.0),"Total integration time")
        ("Nx", po::value<int>()->default_value(100),"Number of grid points in x")
        ("Ny", po::value<int>()->default_value(100),"Number of grid points in y")
        ("ic", po::value<int>()->default_value(1),"Index of the initial condition to use (1-4)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options),vm);
    po::notify(vm);

    // Display help message
    if (vm.count("help"))
    {
        cout << options << endl;
    }

    // Asign parameters
    const double dt = vm["dt"].as<double>();
    const double T = vm["T"].as<double>();
    const int Nx = vm["Nx"].as<int>();
    const int Ny = vm["Ny"].as<int>();
    const int ic = vm["ic"].as<int>();
    
    // debug output
    cout << dt << endl;
    cout << T << endl;
    cout << Nx << endl;
    cout << Ny << endl;
    cout << ic << endl;
    






}