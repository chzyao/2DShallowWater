#include "ShallowWater.h"
#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <cblas.h>

using namespace std;
namespace po = boost::program_options;

ShallowWater::ShallowWater(const po::variables_map &vm)
{
    const double dt = vm["dt"].as<double>();
    const double T = vm["T"].as<double>();
    const int Nx = vm["Nx"].as<int>();
    const int Ny = vm["Ny"].as<int>();
    const int ic = vm["ic"].as<int>();
}

void ShallowWater::SetInitialConditions(double *u, double *v, double *h, double *h0, int Nx, int Ny, int ic, double dx, double dy)
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
                h0[i * Nx + j] = 10.0 + exp(-(i * dx - 50) * (i * dx - 50) / 25.0);
            }
            else if (ic == 2)
            {
                h0[i * Nx + j] = 10.0 + exp(-(j * dy - 50) * (j * dy - 50) / 25.0);
            }
            else if (ic == 3)
            {
                h0[i * Nx + j] = 10.0 + exp(
                                            -((i * dx - 50) * (i * dx - 50) + (j * dy - 50) * (j * dy - 50)) /
                                            25.0);
            }
            else
            {
                h0[i * Nx + j] = 10.0 + exp(-((i * dx - 25) * (i * dx - 25) + (j * dy - 25) * (j * dy - 25)) / 25.0) +
                                 exp(-((i * dx - 75) * (i * dx - 75) + (j * dy - 75) * (j * dy - 75)) / 25.0);
            }
        }
    }

    // copy the initial surface height h0 to h as initial conditions
    cblas_dcopy(Nx * Ny, h0, 1, h, 1);
}

void ShallowWater::SpatialDiscretisation(double *u, int Nx, int Ny, double dx, double dy, char dir, double *deriv)
{
    // Discretisation in x-dir ============================================
    if (dir == 'x')
    {
        double px = 1.0 / dx;

        for (int j = 0; j < Ny; ++j)
        {

            for (int i = 0; i < Nx; ++i)
            {
                deriv[i * Nx + j] =
                    px *
                    (-u[((i - 3 + Nx) % Nx) * Nx + j] / 60.0 + 3.0 / 20.0 * u[((i - 2 + Nx) % Nx) * Nx + j] -
                     3.0 / 4.0 * u[((i - 1 + Nx) % Nx) * Nx + j] + 3.0 / 4.0 * u[((i + 1) % Nx) * Nx + j] -
                     3.0 / 20.0 * u[((i + 2) % Nx) * Nx + j] + u[((i + 3) % Nx) * Nx + j] / 60.0);
            }
        }
    }

    // Discretisation in y-dir ============================================
    else if (dir == 'y')
    {
        double py = 1.0 / dy;

        for (int i = 0; i < Nx; ++i)
        {

            for (int j = 0; j < Ny; ++j)
            {
                deriv[i * Nx + j] =
                    py *
                    (-u[i * Nx + (j - 3 + Ny) % Ny] / 60.0 + 3.0 / 20.0 * u[i * Nx + (j - 2 + Ny) % Ny] -
                     3.0 / 4.0 * u[i * Nx + (j - 1 + Ny) % Ny] + 3.0 / 4.0 * u[i * Nx + (j + 1) % Ny] -
                     3.0 / 20.0 * u[i * Nx + (j + 2) % Ny] + u[i * Nx + (j + 3) % Ny] / 60.0);
            }
        }
    }
}

void ShallowWater::Evaluate_fu(double *u, double *v, double *h, int Nx, int Ny,
                               double dx, double dy, double *f)
{
    double g = 9.81;
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
                            g * derihx[i * Nx + j];
        }
    }

    delete[] deriux;
    delete[] deriuy;
    delete[] derihx;
}

void ShallowWater ::Evaluate_fv(double *u, double *v, double *h, int Nx, int Ny,
                                double dx, double dy, double *f)
{
    double g = 9.81;
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
                            g * derihy[i * Nx + j];
        }
    }

    delete[] derivx;
    delete[] derivy;
    delete[] derihy;
}

void ShallowWater::Evaluate_fh(double *u, double *v, double *h, int Nx, int Ny,
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

void ShallowWater::Evaluate_fu_BLAS(double *u, double *v, double *h, int Nx, int Ny, double dx, double dy, double *f)
{
    double g = 9.81;
    double *deriux = new double[Nx * Ny];
    double *deriuy = new double[Nx * Ny];
    double *derihx = new double[Nx * Ny];

    SpatialDiscretisation(u, Nx, Ny, dx, dy, 'x', deriux);
    SpatialDiscretisation(u, Nx, Ny, dx, dy, 'y', deriuy);
    SpatialDiscretisation(h, Nx, Ny, dx, dy, 'x', derihx);

    cblas_ddot(Nx * Ny, u, 1, deriux, 1);
    cblas_daxpy(Nx * Ny, -1.0, deriux, 1, f, 1);

    cblas_ddot(Nx * Ny, u, 1, deriuy, 1);
    cblas_daxpy(Nx * Ny, -1.0, deriuy, 1, f, 1);

    cblas_ddot(Nx * Ny, h, 1, derihx, 1);
    cblas_daxpy(Nx * Ny, -1.0, derihx, 1, f, 1);
}

void ShallowWater::Evaluate_fv_BLAS(double *u, double *v, double *h, int Nx, int Ny, double dx, double dy, double *f)
{
    double g = 9.81;
    double *derivx = new double[Nx * Ny];
    double *derivy = new double[Nx * Ny];
    double *derihy = new double[Nx * Ny];

    SpatialDiscretisation(v, Nx, Ny, dx, dy, 'x', derivx);
    SpatialDiscretisation(v, Nx, Ny, dx, dy, 'y', derivy);
    SpatialDiscretisation(h, Nx, Ny, dx, dy, 'y', derihy);

    cblas_ddot(Nx * Ny, u, 1, derivx, 1);
    cblas_daxpy(Nx * Ny, -1.0, derivx, 1, f, 1);

    cblas_ddot(Nx * Ny, u, 1, derivy, 1);
    cblas_daxpy(Nx * Ny, -1.0, derivy, 1, f, 1);

    cblas_ddot(Nx * Ny, h, 1, derihy, 1);
    cblas_daxpy(Nx * Ny, -1.0, derihy, 1, f, 1);
}

void ShallowWater::Evaluate_fh_BLAS(double *u, double *v, double *h, int Nx, int Ny, double dx, double dy, double *f)
{
    double *derihux = new double[Nx * Ny];
    double *derihvy = new double[Nx * Ny];
    double *hu = new double[Nx * Ny];
    double *hv = new double[Nx * Ny];

    SpatialDiscretisation(hu, Nx, Ny, dx, dy, 'x', derihux);
    SpatialDiscretisation(hv, Nx, Ny, dx, dy, 'y', derihvy);

    cblas_ddot(Nx * Ny, h, 1, u, 1);
    cblas_dcopy(Nx * Ny, u, 1, hu, 1);
    cblas_daxpy(Nx * Ny, -1.0, hu, 1, f, 1);

    cblas_ddot(Nx * Ny, h, 1, v, 1);
    cblas_dcopy(Nx * Ny, u, 1, hv, 1);
    cblas_daxpy(Nx * Ny, -1.0, hv, 1, f, 1);
}

void ShallowWater::TimeIntegration(double *u, double *v, double *h, int Nx, int Ny, double dx, double dy, double dt, double *fu, double *fv, double *fh)
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
    double *tv = new double[Nx * Ny]; // temp vector t = v
    double *th = new double[Nx * Ny]; // temp vector t = h

    // Calculating k1 = f(yn) ===================================
    cblas_dcopy(Nx * Ny, u, 1, tu, 1);
    cblas_dcopy(Nx * Ny, v, 1, tv, 1);
    cblas_dcopy(Nx * Ny, h, 1, th, 1);

    // Evaluate_fu(u, v, h, Nx, Ny, dx, dy, fu);
    // Evaluate_fv(u, v, h, Nx, Ny, dx, dy, fv);
    // Evaluate_fh(u, v, h, Nx, Ny, dx, dy, fh);

    Evaluate_fu_BLAS(u, v, h, Nx, Ny, dx, dy, fu);
    Evaluate_fv_BLAS(u, v, h, Nx, Ny, dx, dy, fv);
    Evaluate_fh_BLAS(u, v, h, Nx, Ny, dx, dy, fh);

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
    // Evaluate_fu(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh(tu, tv, th, Nx, Ny, dx, dy, fh);

    Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

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

    // Evaluate_fu(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh(tu, tv, th, Nx, Ny, dx, dy, fh);

    Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

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

    // Evaluate_fu(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh(tu, tv, th, Nx, Ny, dx, dy, fh);

    Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

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

ShallowWater::~ShallowWater()
{
}