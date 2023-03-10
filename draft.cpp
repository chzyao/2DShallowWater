#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "ShallowWater.h"
#include "cblas.h"

using namespace std;
namespace po = boost::program_options;

void SetInitialConditions(double *u, double *v, double *h, double *g, int Nx, int Ny, int ic, double dx, double dy)
{
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            // All coded in row-major for now
            u[i * Nx + j] = 0.0;
            v[i * Nx + j] = 0.0;
            if (ic == 1)
            {
                g[i * Nx + j] = exp(-(i * dx - 50) * (i * dx - 50) / 25.0);
            }
            else if (ic == 2)
            {
                g[i * Nx + j] = exp(-(j * dy - 50) * (j * dy - 50) / 25.0);
            }
            else if (ic == 3)
            {
                g[i * Nx + j] = exp(-((i * dx - 50) * (i * dx - 50) + (j * dy - 50) * (j * dy - 50)) / 25.0);
            }
            else
            {
                g[i * Nx + j] = exp(-((i * dx - 25) * (i * dx - 25) + (j * dy - 25) * (j * dy - 25)) / 25.0) + exp(-((i * dx - 75) * (i * dx - 75) + (j * dy - 75) * (j * dy - 75)) / 25.0);
            }
        }
    }

    // copy the initial surface height g to h as initial conditions
    cblas_dcopy(Nx * Ny, g, 1, h, 1);
}

void SpatialDiscretisation(double *u, int Nx, int Ny, double dx, double dy,
                           char dir, double *deriv)
{
    if (dir == 'x')
    {
        double px = 1.0 / dx;

        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                deriv[i * Nx + j] = px * (-u[(i - 3) * Nx + j] / 60.0 + 3.0 / 20.0 * u[(i - 2) * Nx + j] - 3.0 / 4.0 * u[(i - 1) * Nx + j] + 3.0 / 4.0 * u[(i + 1) * Nx + j] - 3.0 / 20.0 * u[(i + 2) * Nx + j] + u[(i + 3) * Nx + j] / 60.0);
            }
        }
    }
    else if (dir == 'y')
    {
        double py = 1.0 / dy;

        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                deriv[i * Nx + j] = py * (-u[i * Nx + j - 3] / 60.0 + 3.0 / 20.0 * u[i * Nx + j - 2] - 3.0 / 4.0 * u[i * Nx + j - 1] + 3.0 / 4.0 * u[i * Nx + j + 1] - 3.0 / 20.0 * u[i * Nx + j + 2] + u[i * Nx + j + 3] / 60.0);
            }
        }
    }
}

void Evaluate_fu(double *u, double *v, double *h, double *g, int Nx, int Ny, double dx, double dy, double *f)
{
    double *deriux = new double[Nx * Ny];
    double *deriuy = new double[Nx * Ny];
    double *derihx = new double[Nx * Ny];

    SpatialDiscretisation(u, Nx, Ny, dx, dy, 'x', deriux);
    SpatialDiscretisation(u, Nx, Ny, dx, dy, 'y', deriuy);
    SpatialDiscretisation(h, Nx, Ny, dx, dy, 'x', derihx);

    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            f[i * Nx + j] = -u[i * Nx + j] * deriux[i * Nx + j] - v[i * Nx + j] * deriuy[i * Nx + j] - g[i * Nx + j] * derihx[i * Nx + j];
        }
    }

    delete[] deriux;
    delete[] deriuy;
    delete[] derihx;
}

void Evaluate_fv(double *u, double *v, double *h, double *g, int Nx, int Ny, double dx, double dy, double *f)
{
    double *derivx = new double[Nx * Ny];
    double *derivy = new double[Nx * Ny];
    double *derihy = new double[Nx * Ny];

    SpatialDiscretisation(v, Nx, Ny, dx, dy, 'x', derivx);
    SpatialDiscretisation(v, Nx, Ny, dx, dy, 'y', derivy);
    SpatialDiscretisation(h, Nx, Ny, dx, dy, 'y', derihy);

    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            f[i * Nx + j] = -u[i * Nx + j] * derivx[i * Nx + j] - v[i * Nx + j] * derivy[i * Nx + j] - g[i * Nx + j] * derihy[i * Nx + j];
        }
    }

    delete[] derivx;
    delete[] derivy;
    delete[] derihy;
}

void Evaluate_fh(double *u, double *v, double *h, double *g, int Nx, int Ny, double dx, double dy, double *f)
{
    double *derihux = new double[Nx * Ny];
    double *derihvy = new double[Nx * Ny];
    double *hu = new double[Nx * Ny];
    double *hv = new double[Nx * Ny];

    // find hu and hv
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            hu[i * Nx + j] = h[i * Nx + j] * u[i * Nx + j];
            hv[i * Nx + j] = h[i * Nx + j] * v[i * Nx + j];
        }
    }

    SpatialDiscretisation(hu, Nx, Ny, dx, dy, 'x', derihux);
    SpatialDiscretisation(hv, Nx, Ny, dx, dy, 'y', derihvy);

    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            f[i * Nx + j] = -derihux[i * Nx + j] - derihvy[i * Nx + j];
        }
    }

    delete[] derihux;
    delete[] derihvy;
    delete[] hu;
    delete[] hv;
}

void TimeIntegrate(double *yn, double dt, int Nx, int Ny)
{
    double *k1 = new double[Nx * Ny];
    double *k2 = new double[Nx * Ny];
    double *k3 = new double[Nx * Ny];
    double *k4 = new double[Nx * Ny];

    // deallocate memory
    delete[] k1;
    delete[] k2;
    delete[] k3;
    delete[] k4;
}

int main(int argc, char *argv[])
{
    cout << "Goodbye World" << endl;
    // Read parameters from command line
    po::options_description options("Available Options.");
    options.add_options()("help", "Display help message")(
        "dt", po::value<double>()->default_value(0.1), "Time-step to use")(
        "T", po::value<double>()->default_value(20.0),
        "Total integration time")("Nx", po::value<int>()->default_value(100),
                                  "Number of grid points in x")(
        "Ny", po::value<int>()->default_value(100),
        "Number of grid points in y")(
        "ic", po::value<int>()->default_value(1),
        "Index of the initial condition to use (1-4)");

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
    double *g = new double[Nx * Ny];

    // debug output
    cout << dt << endl;
    cout << T << endl;
    cout << Nx << endl;
    cout << Ny << endl;
    cout << ic << endl;

    // calculating dx and dy
    const double dx = 1.0;
    const double dy = 1.0;

    // ======================================================
    // test for SetInitialConditions
    SetInitialConditions(u, v, h, g, Nx, Ny, ic, dx, dy);

    // ======================================================
    // test for evaluating f
    double *fu = new double[Nx * Ny];
    double *fv = new double[Nx * Ny];
    double *fh = new double[Nx * Ny];
    Evaluate_fu(u, v, h, g, Nx, Ny, dx, dy, fu);
    Evaluate_fv(u, v, h, g, Nx, Ny, dx, dy, fv);
    Evaluate_fh(u, v, h, g, Nx, Ny, dx, dy, fh);



    // deallocations
    delete[] u;
    delete[] v;
    delete[] h;
}