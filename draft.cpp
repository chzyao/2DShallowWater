#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <iomanip>
#include <fstream>

#include "ShallowWater.h"

using namespace std;
namespace po = boost::program_options;

void SetInitialConditions(double *u, double *v, double *h, int Nx, int Ny, int ic, double dx, double dy)
{
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            // All coded in row-major for now
            u[i * Nx + j] = 0;
            v[i * Nx + j] = 0;
            if (ic == 1)
            {
                h[i * Nx + j] = exp(-(i * dx - 50) * (i * dx - 50) / 25.0);
            }
            else if (ic == 2)
            {
                h[i * Nx + j] = exp(-(j * dy - 50) * (j * dy - 50) / 25.0);
            }
            else if (ic == 3)
            {
                h[i * Nx + j] = exp(-((i * dx - 50) * (i * dx - 50) + (j * dy - 50) * (j * dy - 50)) / 25.0);
            }
            else
            {
                h[i * Nx + j] = exp(-((i * dx - 25) * (i * dx - 25) + (j * dy - 25) * (j * dy - 25)) / 25.0) + exp(-((i * dx - 75) * (i * dx - 75) + (j * dy - 75) * (j * dy - 75)) / 25.0);
            }
        }
    }
}

void printVector(int n, double *b)
{
    for (int i = 0; i < n; ++i)
    {
        cout << b[i] << " ";
    }
    cout << endl;
}

int main(int argc, char *argv[])
{
    cout << "Goodbye World" << endl;
    // Read parameters from command line
    po::options_description options("Available Options.");
    options.add_options()("help", "Display help message")("dt", po::value<double>()->default_value(0.1), "Time-step to use")("T", po::value<double>()->default_value(20.0), "Total integration time")("Nx", po::value<int>()->default_value(100), "Number of grid points in x")("Ny", po::value<int>()->default_value(100), "Number of grid points in y")("ic", po::value<int>()->default_value(1), "Index of the initial condition to use (1-4)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);

    // Display help message
    if (vm.count("help"))
    {
        cout << options << endl;
    }

    // Assign parameters
    const double dt = vm["dt"].as<double>();
    const double T = vm["T"].as<double>();
    const int Nx = vm["Nx"].as<int>();
    const int Ny = vm["Ny"].as<int>();
    const int ic = vm["ic"].as<int>();

    // Allocate solution memories
    double *u = new double[Nx * Ny];
    double *v = new double[Nx * Ny];
    double *h = new double[Nx * Ny];

    // debug output
    cout << dt << endl;
    cout << T << endl;
    cout << Nx << endl;
    cout << Ny << endl;
    cout << ic << endl;

    // calculating dx and dy
    const double dx = 1.0;
    const double dy = 1.0;

    // test for SetInitialConditions
    SetInitialConditions(u, v, h, Nx, Ny, ic, dx, dy);
    cout << "u" << endl;
    printVector(Nx * Ny, u);
    cout << "-----------------------" << endl;
    cout << "v" << endl;
    printVector(Nx * Ny, v);
    cout << "-----------------------" << endl;
    cout << "h" << endl;
    printVector(Nx * Ny, h);

    // deallocations
    delete[] u;
    delete[] v;
    delete[] h;
}