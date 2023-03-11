#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "ShallowWater.h"
#include "cblas.h"

using namespace std;
namespace po = boost::program_options;

void printMatrix(int Nx, int Ny, double *A)
{
    for (int i = 0; i < Nx; i++)
    {
        for (int j = 0; j < Ny; j++)
        {
            cout << A[i * Nx + j] << "  ";
        }
        cout << endl;
    }
}

void SetInitialConditions(double *u, double *v, double *h, double *g, int Nx,
                          int Ny, int ic, double dx, double dy)
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
                g[i * Nx + j] = exp(
                    -((i * dx - 50) * (i * dx - 50) + (j * dy - 50) * (j * dy - 50)) /
                    25.0);
            }
            else
            {
                g[i * Nx + j] = exp(-((i * dx - 25) * (i * dx - 25) +
                                      (j * dy - 25) * (j * dy - 25)) /
                                    25.0) +
                                exp(-((i * dx - 75) * (i * dx - 75) +
                                      (j * dy - 75) * (j * dy - 75)) /
                                    25.0);
            }
        }
    }

    // copy the initial surface height g to h as initial conditions
    cblas_dcopy(Nx * Ny, g, 1, h, 1);
}

void SpatialDiscretisation(double *u, int Nx, int Ny, double dx, double dy,
                           char dir, double *deriv)
{
    // Discretisation in x-dir ============================================
    if (dir == 'x')
    {
        double px = 1.0 / dx;

        for (int j = 0; j < Ny; ++j)
        {
            // Periodic BC i = 0
            deriv[0 * Nx + j] =
                px *
                (-u[(Nx - 3) * Nx + j] / 60.0 + 3.0 / 20.0 * u[(Nx - 2) * Nx + j] -
                 3.0 / 4.0 * u[(Nx - 1) * Nx + j] + 3.0 / 4.0 * u[1 * Nx + j] -
                 3.0 / 20.0 * u[2 * Nx + j] + u[3 * Nx + j] / 60.0);

            // Periodic BC i = 1
            deriv[1 * Nx + j] =
                px *
                (-u[(Nx - 2) * Nx + j] / 60.0 + 3.0 / 20.0 * u[(Nx - 1) * Nx + j] -
                 3.0 / 4.0 * u[0 * Nx + j] + 3.0 / 4.0 * u[2 * Nx + j] -
                 3.0 / 20.0 * u[3 * Nx + j] + u[4 * Nx + j] / 60.0);

            // Periodic BC i = 2
            deriv[2 * Nx + j] =
                px * (-u[(Nx - 1) * Nx + j] / 60.0 + 3.0 / 20.0 * u[0 * Nx + j] -
                      3.0 / 4.0 * u[1 * Nx + j] + 3.0 / 4.0 * u[3 * Nx + j] -
                      3.0 / 20.0 * u[4 * Nx + j] + u[5 * Nx + j] / 60.0);

            // Normal centred scheme
            for (int i = 3; i < Nx - 3; ++i)
            {
                deriv[i * Nx + j] =
                    px *
                    (-u[(i - 3) * Nx + j] / 60.0 + 3.0 / 20.0 * u[(i - 2) * Nx + j] -
                     3.0 / 4.0 * u[(i - 1) * Nx + j] + 3.0 / 4.0 * u[(i + 1) * Nx + j] -
                     3.0 / 20.0 * u[(i + 2) * Nx + j] + u[(i + 3) * Nx + j] / 60.0);
            }

            // Periodic BC i = Nx-3
            deriv[(Nx - 3) * Nx + j] =
                px *
                (-u[(Nx - 6) * Nx + j] / 60.0 + 3.0 / 20.0 * u[(Nx - 5) * Nx + j] -
                 3.0 / 4.0 * u[(Nx - 4) * Nx + j] + 3.0 / 4.0 * u[(Nx - 2) * Nx + j] -
                 3.0 / 20.0 * u[(Nx - 1) * Nx + j] + u[0 * Nx + j] / 60.0);

            // Periodic BC i = Nx-2
            deriv[(Nx - 2) * Nx + j] =
                px *
                (-u[(Nx - 5) * Nx + j] / 60.0 + 3.0 / 20.0 * u[(Nx - 4) * Nx + j] -
                 3.0 / 4.0 * u[(Nx - 3) * Nx + j] + 3.0 / 4.0 * u[(Nx - 1) * Nx + j] -
                 3.0 / 20.0 * u[0 * Nx + j] + u[1 * Nx + j] / 60.0);

            // Periodic BC i = Nx-1
            deriv[(Nx - 1) * Nx + j] =
                px *
                (-u[(Nx - 4) * Nx + j] / 60.0 + 3.0 / 20.0 * u[(Nx - 3) * Nx + j] -
                 3.0 / 4.0 * u[(Nx - 2) * Nx + j] + 3.0 / 4.0 * u[0 * Nx + j] -
                 3.0 / 20.0 * u[1 * Nx + j] + u[2 * Nx + j] / 60.0);
        }
    }

    // Discretisation in y-dir ============================================
    else if (dir == 'y')
    {
        double py = 1.0 / dy;

        for (int i = 0; i < Nx; ++i)
        {
            // Periodic BC for j = 0
            deriv[i * Nx + 0] =
                py * (-u[i * Nx + Ny - 3] / 60.0 + 3.0 / 20.0 * u[i * Nx + Ny - 2] -
                      3.0 / 4.0 * u[i * Nx + Ny - 1] + 3.0 / 4.0 * u[i * Nx + 1] -
                      3.0 / 20.0 * u[i * Nx + 2] + u[i * Nx + 3] / 60.0);

            // Periodic BC for j = 1
            deriv[i * Nx + 1] =
                py * (-u[i * Nx + Ny - 2] / 60.0 + 3.0 / 20.0 * u[i * Nx + Ny - 1] -
                      3.0 / 4.0 * u[i * Nx + 0] + 3.0 / 4.0 * u[i * Nx + 2] -
                      3.0 / 20.0 * u[i * Nx + 3] + u[i * Nx + 4] / 60.0);

            // Periodic BC for j = 2
            deriv[i * Nx + 2] =
                py * (-u[i * Nx + Ny - 1] / 60.0 + 3.0 / 20.0 * u[i * Nx + 0] -
                      3.0 / 4.0 * u[i * Nx + 1] + 3.0 / 4.0 * u[i * Nx + 3] -
                      3.0 / 20.0 * u[i * Nx + 4] + u[i * Nx + 5] / 60.0);

            // Normal centred scheme
            for (int j = 3; j < Ny - 3; ++j)
            {
                deriv[i * Nx + j] =
                    py *
                    (-u[i * Nx + j - 3] / 60.0 + 3.0 / 20.0 * u[i * Nx + j - 2] -
                     3.0 / 4.0 * u[i * Nx + j - 1] + 3.0 / 4.0 * u[i * Nx + j + 1] -
                     3.0 / 20.0 * u[i * Nx + j + 2] + u[i * Nx + j + 3] / 60.0);
            }

            // Periodic BC for j = Ny-3
            deriv[i * Nx + Ny - 3] =
                py *
                (-u[i * Nx + Ny - 6] / 60.0 + 3.0 / 20.0 * u[i * Nx + Ny - 5] -
                 3.0 / 4.0 * u[i * Nx + Ny - 4] + 3.0 / 4.0 * u[i * Nx + Ny - 2] -
                 3.0 / 20.0 * u[i * Nx + Ny - 1] + u[i * Nx + Ny] / 60.0);

            // Periodic BC for j = Ny-2
            deriv[i * Nx + Ny - 2] =
                py *
                (-u[i * Nx + Ny - 5] / 60.0 + 3.0 / 20.0 * u[i * Nx + Ny - 4] -
                 3.0 / 4.0 * u[i * Nx + Ny - 3] + 3.0 / 4.0 * u[i * Nx + Ny - 1] -
                 3.0 / 20.0 * u[i * Nx + 0] + u[i * Nx + 1] / 60.0);

            // Periodic BC for j = Ny-1
            deriv[i * Nx + Ny - 1] =
                py * (-u[i * Nx + Ny - 4] / 60.0 + 3.0 / 20.0 * u[i * Nx + Ny - 3] -
                      3.0 / 4.0 * u[i * Nx + Ny - 2] + 3.0 / 4.0 * u[i * Nx + 0] -
                      3.0 / 20.0 * u[i * Nx + 1] + u[i * Nx + 2] / 60.0);
        }
    }
}

void Evaluate_fu(double *u, double *v, double *h, double *g, int Nx, int Ny,
                 double dx, double dy, double *f)
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
            f[i * Nx + j] = -u[i * Nx + j] * deriux[i * Nx + j] -
                            v[i * Nx + j] * deriuy[i * Nx + j] -
                            g[i * Nx + j] * derihx[i * Nx + j];
        }
    }

    delete[] deriux;
    delete[] deriuy;
    delete[] derihx;
}

void Evaluate_fv(double *u, double *v, double *h, double *g, int Nx, int Ny,
                 double dx, double dy, double *f)
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
            f[i * Nx + j] = -u[i * Nx + j] * derivx[i * Nx + j] -
                            v[i * Nx + j] * derivy[i * Nx + j] -
                            g[i * Nx + j] * derihy[i * Nx + j];
        }
    }

    delete[] derivx;
    delete[] derivy;
    delete[] derihy;
}

void Evaluate_fh(double *u, double *v, double *h, double *g, int Nx, int Ny,
                 double dx, double dy, double *f)
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

void TimeIntegration(double *u, double *v, double *h, double *g, int Nx, int Ny,
                     double dx, double dy, double dt, double T, double *fu,
                     double *fv, double *fh)
{
    // Solving for u
    double *k1_u = new double[Nx * Ny];
    double *k2_u = new double[Nx * Ny];
    double *k3_u = new double[Nx * Ny];
    double *k4_u = new double[Nx * Ny];

    // Solve for v
    double *k1_v = new double[Nx * Ny];
    double *k2_v = new double[Nx * Ny];
    double *k3_v = new double[Nx * Ny];
    double *k4_v = new double[Nx * Ny];

    // Solve for h
    double *k1_h = new double[Nx * Ny];
    double *k2_h = new double[Nx * Ny];
    double *k3_h = new double[Nx * Ny];
    double *k4_h = new double[Nx * Ny];

    double *tu = new double[Nx * Ny]; // temp vector t = u
    double *tv = new double[Nx * Ny]; // temp vector t = u
    double *th = new double[Nx * Ny]; // temp vector t = u

    // Calculating k1 = f(yn) ===================================
    cblas_dcopy(Nx * Ny, u, 1, tu, 1);
    cblas_dcopy(Nx * Ny, v, 1, tv, 1);
    cblas_dcopy(Nx * Ny, h, 1, th, 1);

    Evaluate_fu(u, v, h, g, Nx, Ny, dx, dy, fu);
    Evaluate_fv(u, v, h, g, Nx, Ny, dx, dy, fv);
    Evaluate_fh(u, v, h, g, Nx, Ny, dx, dy, fh);

    cblas_dcopy(Nx * Ny, fu, 1, k1_u, 1);
    cblas_dcopy(Nx * Ny, fv, 1, k1_v, 1);
    cblas_dcopy(Nx * Ny, fh, 1, k1_h, 1);

    // Calculating k2 = f(yn + dt*k1/2) ==========================
    // reset temp values
    cblas_dcopy(Nx * Ny, u, 1, tu, 1); // reset tu to u
    cblas_dcopy(Nx * Ny, v, 1, tv, 1);
    cblas_dcopy(Nx * Ny, h, 1, th, 1);

    // update un to un+dt*k1/2 to evaluate f for k2
    cblas_daxpy(Nx * Ny, dt / 2.0, k1_u, 1, tu, 1);
    cblas_daxpy(Nx * Ny, dt / 2.0, k1_v, 1, tv, 1);
    cblas_daxpy(Nx * Ny, dt / 2.0, k1_h, 1, th, 1);

    // Evaluate new f
    Evaluate_fu(tu, v, h, g, Nx, Ny, dx, dy, fu);
    Evaluate_fv(tv, v, h, g, Nx, Ny, dx, dy, fv);
    Evaluate_fh(th, v, h, g, Nx, Ny, dx, dy, fh);

    cblas_dcopy(Nx * Ny, fu, 1, k2_u, 1);
    cblas_dcopy(Nx * Ny, fv, 1, k2_v, 1);
    cblas_dcopy(Nx * Ny, fh, 1, k2_h, 1);

    // Calculating k3 = f(yn+dt*k2/2) =============================
    // reset temp values
    cblas_dcopy(Nx * Ny, u, 1, tu, 1); // reset tu to u
    cblas_dcopy(Nx * Ny, v, 1, tv, 1);
    cblas_dcopy(Nx * Ny, h, 1, th, 1);

    // update un to un+dt*k2/2 to evaluate f for k3
    cblas_daxpy(Nx * Ny, dt / 2.0, k2_u, 1, tu, 1);
    cblas_daxpy(Nx * Ny, dt / 2.0, k2_v, 1, tv, 1);
    cblas_daxpy(Nx * Ny, dt / 2.0, k2_h, 1, th, 1);

    Evaluate_fu(tu, v, h, g, Nx, Ny, dx, dy, fu);
    Evaluate_fv(tv, v, h, g, Nx, Ny, dx, dy, fv);
    Evaluate_fh(th, v, h, g, Nx, Ny, dx, dy, fh);

    cblas_dcopy(Nx * Ny, fu, 1, k3_u, 1);
    cblas_dcopy(Nx * Ny, fv, 1, k3_v, 1);
    cblas_dcopy(Nx * Ny, fh, 1, k3_h, 1);

    // k4 = f(yn+dt*k3) ===========================================
    // reset temp values
    cblas_dcopy(Nx * Ny, u, 1, tu, 1); // reset tu to u
    cblas_dcopy(Nx * Ny, v, 1, tv, 1);
    cblas_dcopy(Nx * Ny, h, 1, th, 1);

    // update un to un+dt*k2/2 to evaluate f for k3
    cblas_daxpy(Nx * Ny, dt, k3_u, 1, tu, 1);
    cblas_daxpy(Nx * Ny, dt, k3_v, 1, tv, 1);
    cblas_daxpy(Nx * Ny, dt, k3_h, 1, th, 1);

    Evaluate_fu(tu, v, h, g, Nx, Ny, dx, dy, fu);
    Evaluate_fv(tv, v, h, g, Nx, Ny, dx, dy, fv);
    Evaluate_fh(th, v, h, g, Nx, Ny, dx, dy, fh);

    cblas_dcopy(Nx * Ny, fu, 1, k4_u, 1);
    cblas_dcopy(Nx * Ny, fv, 1, k4_v, 1);
    cblas_dcopy(Nx * Ny, fh, 1, k4_h, 1);

    // yn+1 = yn + 1/6*(k1+2*k2+2*k3+k4)*dt
    // Update solution
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            u[i * Nx + j] += dt / 6.0 *
                             (k1_u[i * Nx + j] + 2.0 * k2_u[i * Nx + j] +
                              2.0 * k3_u[i * Nx + j] + k4_u[i * Nx + j]);
            v[i * Nx + j] += dt / 6.0 *
                             (k1_v[i * Nx + j] + 2.0 * k2_v[i * Nx + j] +
                              2.0 * k3_v[i * Nx + j] + k4_v[i * Nx + j]);
            h[i * Nx + j] += dt / 6.0 *
                             (k1_h[i * Nx + j] + 2.0 * k2_h[i * Nx + j] +
                              2.0 * k3_h[i * Nx + j] + k4_h[i * Nx + j]);
        }
    }

    // deallocate memory
    delete[] k1_u;
    delete[] k2_u;
    delete[] k3_u;
    delete[] k4_u;

    delete[] k1_v;
    delete[] k2_v;
    delete[] k3_v;
    delete[] k4_v;

    delete[] k1_h;
    delete[] k2_h;
    delete[] k3_h;
    delete[] k4_h;
}

int main(int argc, char *argv[])
{
    cout << "Goodbye World" << endl;
    // Read parameters from command line
    po::options_description options("Available Options.");
    options.add_options()("help", "Display help message")(
        "dt", po::value<double>()->default_value(0.1), "Time-step to use")(
        "T", po::value<double>()->default_value(20.0), "Total integration time")(
        "Nx", po::value<int>()->default_value(100), "Number of grid points in x")(
        "Ny", po::value<int>()->default_value(100), "Number of grid points in y")(
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

    // verify outputs
    cout << "fu" << endl;
    printMatrix(Nx, Ny, fu);
    cout << "fv" << endl;
    printMatrix(Nx, Ny, fv);
    cout << "fh" << endl;
    printMatrix(Nx, Ny, fh);

    // ======================================================
    // 4th order RK Time Integrations

    // Time advancement
    double time = 0.0; // start time
    while (time <= T)
    {
        TimeIntegration(u, v, h, g, Nx, Ny, dx, dy, dt, T, fu, fv, fh);
        time += dt;
    }

    // deallocations
    delete[] u;
    delete[] v;
    delete[] h;
    delete[] g;
    delete[] fu;
    delete[] fv;
    delete[] fh;
}