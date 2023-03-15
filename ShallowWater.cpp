#include "ShallowWater.h"
#include <iostream>
#include <boost/program_options.hpp>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <cblas.h>

using namespace std;
namespace po = boost::program_options;

ShallowWater::ShallowWater()
    : m_dt(0.1), m_T(80.0), m_Nx(100), m_Ny(100), m_ic(3)
{
}

void ShallowWater::SetParameters(int argc, char *argv[])
{
    // Read parameters from command line =========================
    po::options_description options("Available Options.");
    options.add_options()("help", "Display help message")(
        "dt", po::value<double>()->default_value(0.1), "Time-step to use")(
        "T", po::value<double>()->default_value(80.0), "Total integration time")(
        "Nx", po::value<int>()->default_value(100), "Number of grid points in x")(
        "Ny", po::value<int>()->default_value(100), "Number of grid points in y")(
        "ic", po::value<int>()->default_value(3),
        "Index of the initial condition to use (1-4)");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);

    // Display help message
    if (vm.count("help"))
    {
        cout << options << endl;
        exit(EXIT_SUCCESS);
    }

    // Assign parameters
    m_dt = vm["dt"].as<double>();
    m_T = vm["T"].as<double>();
    m_Nx = vm["Nx"].as<int>();
    m_Ny = vm["Ny"].as<int>();
    m_ic = vm["ic"].as<int>();

    // Memory Allocation for solutions
    m_u = new double[m_Nx * m_Ny];
    m_v = new double[m_Nx * m_Ny];
    m_h = new double[m_Nx * m_Ny];
    m_h0 = new double[m_Nx * m_Ny];
}

void ShallowWater::SetInitialConditions(double *u, double *v, double *h)
{
    for (int i = 0; i < m_Nx; ++i)
    {
        for (int j = 0; j < m_Ny; ++j)
        {
            // All coded in row-major for now
            m_u[i * m_Nx + j] = 0.0;
            m_v[i * m_Nx + j] = 0.0;
            if (m_ic == 1)
            {
                m_h0[i * m_Nx + j] = 10.0 + exp(-(i * m_dx - 50) * (i * m_dx - 50) / 25.0);
            }
            else if (m_ic == 2)
            {
                m_h0[i * m_Nx + j] = 10.0 + exp(-(j * m_dy - 50) * (j * m_dy - 50) / 25.0);
            }
            else if (m_ic == 3)
            {
                m_h0[i * m_Nx + j] = 10.0 + exp(
                                                -((i * m_dx - 50) * (i * m_dx - 50) + (j * m_dy - 50) * (j * m_dy - 50)) /
                                                25.0);
            }
            else
            {
                m_h0[i * m_Nx + j] = 10.0 + exp(-((i * m_dx - 25) * (i * m_dx - 25) + (j * m_dy - 25) * (j * m_dy - 25)) / 25.0) +
                                     exp(-((i * m_dx - 75) * (i * m_dx - 75) + (j * m_dy - 75) * (j * m_dy - 75)) / 25.0);
            }
        }
    }

    // copy the initial surface height h0 to h as initial conditions
    cblas_dcopy(m_Nx * m_Ny, m_h0, 1, m_h, 1);
}

void ShallowWater::SpatialDiscretisation(double *u, char dir, double *deriv)
{
    // Discretisation in x-dir ============================================
    if (dir == 'x')
    {
        double px = 1.0 / m_dx;

        for (int j = 0; j < m_Ny; ++j)
        {

            for (int i = 0; i < m_Nx; ++i)
            {
                deriv[i * m_Nx + j] =
                    px *
                    (-u[((i - 3 + m_Nx) % m_Nx) * m_Nx + j] / 60.0 + 3.0 / 20.0 * u[((i - 2 + m_Nx) % m_Nx) * m_Nx + j] -
                     3.0 / 4.0 * u[((i - 1 + m_Nx) % m_Nx) * m_Nx + j] + 3.0 / 4.0 * u[((i + 1) % m_Nx) * m_Nx + j] -
                     3.0 / 20.0 * u[((i + 2) % m_Nx) * m_Nx + j] + u[((i + 3) % m_Nx) * m_Nx + j] / 60.0);
            }
        }
    }

    // Discretisation in y-dir ============================================
    else if (dir == 'y')
    {
        double py = 1.0 / m_dy;

        for (int i = 0; i < m_Nx; ++i)
        {

            for (int j = 0; j < m_Ny; ++j)
            {
                deriv[i * m_Nx + j] =
                    py *
                    (-u[i * m_Nx + (j - 3 + m_Ny) % m_Ny] / 60.0 + 3.0 / 20.0 * u[i * m_Nx + (j - 2 + m_Ny) % m_Ny] -
                     3.0 / 4.0 * u[i * m_Nx + (j - 1 + m_Ny) % m_Ny] + 3.0 / 4.0 * u[i * m_Nx + (j + 1) % m_Ny] -
                     3.0 / 20.0 * u[i * m_Nx + (j + 2) % m_Ny] + u[i * m_Nx + (j + 3) % m_Ny] / 60.0);
            }
        }
    }
}

void ShallowWater::Evaluate_fu(double *u, double *v, double *h, double *f)
{
    double g = 9.81;
    double *deriux = new double[m_Nx * m_Ny];
    double *deriuy = new double[m_Nx * m_Ny];
    double *derihx = new double[m_Nx * m_Ny];

    SpatialDiscretisation(u, 'x', deriux);
    SpatialDiscretisation(u, 'y', deriuy);
    SpatialDiscretisation(h, 'x', derihx);

    for (int i = 0; i < m_Nx; ++i)
    {
        for (int j = 0; j < m_Ny; ++j)
        {
            f[i * m_Nx + j] = -m_u[i * m_Nx + j] * deriux[i * m_Nx + j] -
                              m_v[i * m_Nx + j] * deriuy[i * m_Nx + j] -
                              g * derihx[i * m_Nx + j];
        }
    }

    delete[] deriux;
    delete[] deriuy;
    delete[] derihx;
}

void ShallowWater::Evaluate_fv(double *u, double *v, double *h, double *f)
{
    double g = 9.81;
    double *derivx = new double[m_Nx * m_Ny];
    double *derivy = new double[m_Nx * m_Ny];
    double *derihy = new double[m_Nx * m_Ny];

    SpatialDiscretisation(v, 'x', derivx);
    SpatialDiscretisation(v, 'y', derivy);
    SpatialDiscretisation(h, 'y', derihy);

    for (int i = 0; i < m_Nx; ++i)
    {
        for (int j = 0; j < m_Ny; ++j)
        {
            f[i * m_Nx + j] = -u[i * m_Nx + j] * derivx[i * m_Nx + j] -
                              v[i * m_Nx + j] * derivy[i * m_Nx + j] -
                              g * derihy[i * m_Nx + j];
        }
    }

    delete[] derivx;
    delete[] derivy;
    delete[] derihy;
}

void ShallowWater::Evaluate_fh(double *u, double *v, double *h, double *f)
{
    double *derihux = new double[m_Nx * m_Ny];
    double *derihvy = new double[m_Nx * m_Ny];
    double *hu = new double[m_Nx * m_Ny];
    double *hv = new double[m_Nx * m_Ny];

    // find hu and hv
    for (int i = 0; i < m_Nx; ++i)
    {
        for (int j = 0; j < m_Ny; ++j)
        {
            hu[i * m_Nx + j] = h[i * m_Nx + j] * u[i * m_Nx + j];
            hv[i * m_Nx + j] = h[i * m_Nx + j] * v[i * m_Nx + j];
        }
    }

    SpatialDiscretisation(hu, 'x', derihux);
    SpatialDiscretisation(hv, 'y', derihvy);

    for (int i = 0; i < m_Nx; ++i)
    {
        for (int j = 0; j < m_Ny; ++j)
        {
            f[i * m_Nx + j] = -derihux[i * m_Nx + j] - derihvy[i * m_Nx + j];
        }
    }

    delete[] derihux;
    delete[] derihvy;
    delete[] hu;
    delete[] hv;
}

// void ShallowWater::Evaluate_fu_BLAS(double *u, double *v, double *h, int Nx, int Ny, double dx, double dy, double *f)
// {
//     double g = 9.81;
//     double *deriux = new double[Nx * Ny];
//     double *deriuy = new double[Nx * Ny];
//     double *derihx = new double[Nx * Ny];

//     SpatialDiscretisation(u, Nx, Ny, dx, dy, 'x', deriux);
//     SpatialDiscretisation(u, Nx, Ny, dx, dy, 'y', deriuy);
//     SpatialDiscretisation(h, Nx, Ny, dx, dy, 'x', derihx);

//     cblas_ddot(Nx * Ny, u, 1, deriux, 1);
//     cblas_daxpy(Nx * Ny, -1.0, deriux, 1, f, 1);

//     cblas_ddot(Nx * Ny, u, 1, deriuy, 1);
//     cblas_daxpy(Nx * Ny, -1.0, deriuy, 1, f, 1);

//     cblas_ddot(Nx * Ny, h, 1, derihx, 1);
//     cblas_daxpy(Nx * Ny, -1.0, derihx, 1, f, 1);
// }

// void ShallowWater::Evaluate_fv_BLAS(double *u, double *v, double *h, int Nx, int Ny, double dx, double dy, double *f)
// {
//     double g = 9.81;
//     double *derivx = new double[Nx * Ny];
//     double *derivy = new double[Nx * Ny];
//     double *derihy = new double[Nx * Ny];

//     SpatialDiscretisation(v, Nx, Ny, dx, dy, 'x', derivx);
//     SpatialDiscretisation(v, Nx, Ny, dx, dy, 'y', derivy);
//     SpatialDiscretisation(h, Nx, Ny, dx, dy, 'y', derihy);

//     cblas_ddot(Nx * Ny, u, 1, derivx, 1);
//     cblas_daxpy(Nx * Ny, -1.0, derivx, 1, f, 1);

//     cblas_ddot(Nx * Ny, u, 1, derivy, 1);
//     cblas_daxpy(Nx * Ny, -1.0, derivy, 1, f, 1);

//     cblas_ddot(Nx * Ny, h, 1, derihy, 1);
//     cblas_daxpy(Nx * Ny, -1.0, derihy, 1, f, 1);
// }

// void ShallowWater::Evaluate_fh_BLAS(double *u, double *v, double *h, int Nx, int Ny, double dx, double dy, double *f)
// {
//     double *derihux = new double[Nx * Ny];
//     double *derihvy = new double[Nx * Ny];
//     double *hu = new double[Nx * Ny];
//     double *hv = new double[Nx * Ny];

//     SpatialDiscretisation(hu, Nx, Ny, dx, dy, 'x', derihux);
//     SpatialDiscretisation(hv, Nx, Ny, dx, dy, 'y', derihvy);

//     cblas_ddot(Nx * Ny, h, 1, u, 1);
//     cblas_dcopy(Nx * Ny, u, 1, hu, 1);
//     cblas_daxpy(Nx * Ny, -1.0, hu, 1, f, 1);

//     cblas_ddot(Nx * Ny, h, 1, v, 1);
//     cblas_dcopy(Nx * Ny, u, 1, hv, 1);
//     cblas_daxpy(Nx * Ny, -1.0, hv, 1, f, 1);
// }

void ShallowWater::TimeIntegration(double *u, double *v, double *h, double *fu, double *fv, double *fh)
{
    // Solving for u
    double *k1_u = new double[m_Nx * m_Ny];
    double *k2_u = new double[m_Nx * m_Ny];
    double *k3_u = new double[m_Nx * m_Ny];
    double *k4_u = new double[m_Nx * m_Ny];

    // Solve for v
    double *k1_v = new double[m_Nx * m_Ny];
    double *k2_v = new double[m_Nx * m_Ny];
    double *k3_v = new double[m_Nx * m_Ny];
    double *k4_v = new double[m_Nx * m_Ny];

    // Solve for h
    double *k1_h = new double[m_Nx * m_Ny];
    double *k2_h = new double[m_Nx * m_Ny];
    double *k3_h = new double[m_Nx * m_Ny];
    double *k4_h = new double[m_Nx * m_Ny];

    double *tu = new double[m_Nx * m_Ny]; // temp vector t = u
    double *tv = new double[m_Nx * m_Ny]; // temp vector t = v
    double *th = new double[m_Nx * m_Ny]; // temp vector t = h

    // Calculating k1 = f(yn) ===================================
    cblas_dcopy(m_Nx * m_Ny, u, 1, tu, 1);
    cblas_dcopy(m_Nx * m_Ny, v, 1, tv, 1);
    cblas_dcopy(m_Nx * m_Ny, h, 1, th, 1);

    Evaluate_fu(u, v, h, fu);
    Evaluate_fv(u, v, h, fv);
    Evaluate_fh(u, v, h, fh);

    // Evaluate_fu_BLAS(u, v, h, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(u, v, h, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(u, v, h, Nx, Ny, dx, dy, fh);

    cblas_dcopy(m_Nx * m_Ny, fu, 1, k1_u, 1);
    cblas_dcopy(m_Nx * m_Ny, fv, 1, k1_v, 1);
    cblas_dcopy(m_Nx * m_Ny, fh, 1, k1_h, 1);

    // Calculating k2 = f(yn + dt*k1/2) ==========================
    // reset temp values
    cblas_dcopy(m_Nx * m_Ny, m_u, 1, tu, 1); // reset tu to u
    cblas_dcopy(m_Nx * m_Ny, m_v, 1, tv, 1);
    cblas_dcopy(m_Nx * m_Ny, m_h, 1, th, 1);

    // update un to un+dt*k1/2 to evaluate f for k2
    cblas_daxpy(m_Nx * m_Ny, m_dt / 2.0, k1_u, 1, tu, 1);
    cblas_daxpy(m_Nx * m_Ny, m_dt / 2.0, k1_v, 1, tv, 1);
    cblas_daxpy(m_Nx * m_Ny, m_dt / 2.0, k1_h, 1, th, 1);

    // Evaluate new f
    Evaluate_fu(tu, tv, th, fu);
    Evaluate_fv(tu, tv, th, fv);
    Evaluate_fh(tu, tv, th, fh);

    // Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

    cblas_dcopy(m_Nx * m_Ny, fu, 1, k2_u, 1);
    cblas_dcopy(m_Nx * m_Ny, fv, 1, k2_v, 1);
    cblas_dcopy(m_Nx * m_Ny, fh, 1, k2_h, 1);

    // Calculating k3 = f(yn+dt*k2/2) =============================
    // reset temp values
    cblas_dcopy(m_Nx * m_Ny, u, 1, tu, 1); // reset tu to u
    cblas_dcopy(m_Nx * m_Ny, v, 1, tv, 1);
    cblas_dcopy(m_Nx * m_Ny, h, 1, th, 1);

    // update un to un+dt*k2/2 to evaluate f for k3
    cblas_daxpy(m_Nx * m_Ny, m_dt / 2.0, k2_u, 1, tu, 1);
    cblas_daxpy(m_Nx * m_Ny, m_dt / 2.0, k2_v, 1, tv, 1);
    cblas_daxpy(m_Nx * m_Ny, m_dt / 2.0, k2_h, 1, th, 1);

    Evaluate_fu(tu, tv, th, fu);
    Evaluate_fv(tu, tv, th, fv);
    Evaluate_fh(tu, tv, th, fh);

    // Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

    cblas_dcopy(m_Nx * m_Ny, fu, 1, k3_u, 1);
    cblas_dcopy(m_Nx * m_Ny, fv, 1, k3_v, 1);
    cblas_dcopy(m_Nx * m_Ny, fh, 1, k3_h, 1);

    // k4 = f(yn+dt*k3) ===========================================
    // reset temp values
    cblas_dcopy(m_Nx * m_Ny, u, 1, tu, 1); // reset tu to u
    cblas_dcopy(m_Nx * m_Ny, v, 1, tv, 1);
    cblas_dcopy(m_Nx * m_Ny, h, 1, th, 1);

    // update un to un+dt*k2/2 to evaluate f for k3
    cblas_daxpy(m_Nx * m_Ny, m_dt, k3_u, 1, tu, 1);
    cblas_daxpy(m_Nx * m_Ny, m_dt, k3_v, 1, tv, 1);
    cblas_daxpy(m_Nx * m_Ny, m_dt, k3_h, 1, th, 1);

    Evaluate_fu(tu, tv, th, fu);
    Evaluate_fv(tu, tv, th, fv);
    Evaluate_fh(tu, tv, th, fh);

    // Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

    cblas_dcopy(m_Nx * m_Ny, fu, 1, k4_u, 1);
    cblas_dcopy(m_Nx * m_Ny, fv, 1, k4_v, 1);
    cblas_dcopy(m_Nx * m_Ny, fh, 1, k4_h, 1);

    // yn+1 = yn + 1/6*(k1+2*k2+2*k3+k4)*dt
    // Update solution
    for (int i = 0; i < m_Nx; ++i)
    {
        for (int j = 0; j < m_Ny; ++j)
        {
            u[i * m_Nx + j] += m_dt / 6.0 *
                             (k1_u[i * m_Nx + j] + 2.0 * k2_u[i * m_Nx + j] +
                              2.0 * k3_u[i * m_Nx + j] + k4_u[i * m_Nx + j]);
            v[i * m_Nx + j] += m_dt / 6.0 *
                             (k1_v[i * m_Nx + j] + 2.0 * k2_v[i * m_Nx + j] +
                              2.0 * k3_v[i * m_Nx + j] + k4_v[i * m_Nx + j]);
            h[i * m_Nx + j] += m_dt / 6.0 *
                             (k1_h[i * m_Nx + j] + 2.0 * k2_h[i * m_Nx + j] +
                              2.0 * k3_h[i * m_Nx + j] + k4_h[i * m_Nx + j]);
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

void ShallowWater::Solve()
{
    // Memory Allocation for solutions
    double *u = new double[m_Nx * m_Ny];
    double *v = new double[m_Nx * m_Ny];
    double *h = new double[m_Nx * m_Ny];

    const double dx = 1.0;
    const double dy = 1.0;

    double *fu = new double[m_Nx * m_Ny];
    double *fv = new double[m_Nx * m_Ny];
    double *fh = new double[m_Nx * m_Ny];

    //  =====================================================
    // Set Initial conditions
    SetInitialConditions(u, v, h);

    // ======================================================
    // 4th order RK Time Integrations

    // Time advancement
    double time = 0.0; // start time
    while (time <= m_T)
    {
        TimeIntegration(u, v, h, fu, fv, fh);
        time += m_dt;
    }

    // ======================================================
    // Write to file
    // Write initial condition
    ofstream vOut("output.txt", ios::out | ios ::trunc);
    vOut.precision(5);
    for (int j = 0; j < m_Ny; ++j)
    {
        for (int i = 0; i < m_Nx; ++i)
        {
            vOut << setw(15) << i * dx << setw(15) << j * dy << setw(15) << u[i * m_Nx + j] << setw(15) << v[i * m_Nx + j] << setw(15) << h[i * m_Nx + j] << endl;
        }
    }

}

ShallowWater::~ShallowWater()
{
}