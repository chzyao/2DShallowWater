#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <cblas.h>

#include "ShallowWater.h"

using namespace std;
namespace po = boost::program_options;


void SetInitialConditions(double *u, double *v, double *h, double *h0, int Nx,
                          int Ny, int ic, double dx, double dy)
{
    for (int i = 0; i < Ny; ++i)
    {
        for (int j = 0; j < Nx; ++j)
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

void SpatialDiscretisation(double *u, int Nx, int Ny, double dx, double dy,
                           char dir, double *deriv)
{
    // Discretisation in x-dir ============================================
    if (dir == 'x')
    {
        double px = 1.0 / dx;

        for (int j = 0; j < Nx; ++j)
        {

            for (int i = 0; i < Ny; ++i)
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

        for (int i = 0; i < Ny; ++i)
        {

            for (int j = 0; j < Nx; ++j)
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


void Evaluate_fu(double *u, double *v, double *h, int Nx, int Ny,
                 double dx, double dy, double *f)
{
    double g = 9.81;
    double *deriux = new double[Nx * Ny];
    double *deriuy = new double[Nx * Ny];
    double *derihx = new double[Nx * Ny];

    SpatialDiscretisation(u, Nx, Ny, dx, dy, 'x', deriux);
    SpatialDiscretisation(u, Nx, Ny, dx, dy, 'y', deriuy);
    SpatialDiscretisation(h, Nx, Ny, dx, dy, 'x', derihx);

    for (int i = 0; i < Ny; ++i)
    {
        for (int j = 0; j < Nx; ++j)
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

void Evaluate_fv(double *u, double *v, double *h, int Nx, int Ny,
                 double dx, double dy, double *f)
{
    double g = 9.81;
    double *derivx = new double[Nx * Ny];
    double *derivy = new double[Nx * Ny];
    double *derihy = new double[Nx * Ny];

    SpatialDiscretisation(v, Nx, Ny, dx, dy, 'x', derivx);
    SpatialDiscretisation(v, Nx, Ny, dx, dy, 'y', derivy);
    SpatialDiscretisation(h, Nx, Ny, dx, dy, 'y', derihy);

    for (int i = 0; i < Ny; ++i)
    {
        for (int j = 0; j < Nx; ++j)
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

void Evaluate_fh(double *u, double *v, double *h, int Nx, int Ny,
                 double dx, double dy, double *f)
{
    double *derihux = new double[Nx * Ny];
    double *derihvy = new double[Nx * Ny];
    double *hu = new double[Nx * Ny];
    double *hv = new double[Nx * Ny];

    // find hu and hv
    for (int i = 0; i < Ny; ++i)
    {
        for (int j = 0; j < Nx; ++j)
        {
            hu[i * Nx + j] = h[i * Nx + j] * u[i * Nx + j];
            hv[i * Nx + j] = h[i * Nx + j] * v[i * Nx + j];
        }
    }

    SpatialDiscretisation(hu, Nx, Ny, dx, dy, 'x', derihux);
    SpatialDiscretisation(hv, Nx, Ny, dx, dy, 'y', derihvy);

    for (int i = 0; i < Ny; ++i)
    {
        for (int j = 0; j < Nx; ++j)
        {
            f[i * Nx + j] = -derihux[i * Nx + j] - derihvy[i * Nx + j];
        }
    }

    delete[] derihux;
    delete[] derihvy;
    delete[] hu;
    delete[] hv;
}


int main(int argc, char *argv[])
{

    // Read parameters from command line =========================

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

        std::cout << options << std::endl;
    }

    // Assign parameters
    const double dt = vm["dt"].as<double>();
    const double T = vm["T"].as<double>();
    const int Nx = vm["Nx"].as<int>();
    const int Ny = vm["Ny"].as<int>();
    const int ic = vm["ic"].as<int>();

    // calculating dx and dy
    const double dx = 1.0;
    const double dy = 1.0;

    // MPI =====================================================
    MPI_Init(&argc, &argv);
    const int root = 0; // root rank

    // Get size and rank
    int world_rank, world_size, retval_rank, retval_size;
    retval_rank = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    retval_size = MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (retval_rank == MPI_ERR_COMM || retval_size == MPI_ERR_COMM)
    {
        std::cout << "Invalid communicator" << std::endl;
        return 1;
    }

    std::cout << "Goodbye World" << std::endl;



    // Subdomain ===============================================
    int Nx_loc = Nx / world_size; // 3 ghost cells each side



    // std::cout << "Nx_loc" << Nx_loc << std::endl;
    // std::cout << "Ny_loc" << Ny_loc << std::endl;

    // // Allocate solution memories (allocated with ghost cells)
    // double *u_loc = new double[Nx_loc * Ny_loc];
    // double *v_loc = new double[Nx_loc * Ny_loc];
    // double *h_loc = new double[Nx_loc * Ny_loc];
    // double *h0_loc = new double[Nx_loc * Ny_loc];

    // double *fu_loc = new double[Nx_loc * Ny_loc];
    // double *fv_loc = new double[Nx_loc * Ny_loc];
    // double *fh_loc = new double[Nx_loc * Ny_loc];

    // // solutions
    // double *u = nullptr;
    // double *v = nullptr;
    // double *h = nullptr;

    // // =========================================================
    // // test for SetInitialConditions
    // SetInitialConditions(u_loc, v_loc, h_loc, h0_loc, Nx_loc, Ny_loc,
    //                      ic, dx, dy, &local_mpi_info);

    // // ======================================================
    // // 4th order RK Time Integrations

    // // Time advancement
    // double time = 0.0; // start time
    // while (time <= T)
    // {
    //     TimeIntegration(u_loc, v_loc, h_loc, Nx_loc, Ny_loc,
    //                     dx, dy, dt, fu_loc, fv_loc, fh_loc, &local_mpi_info);
    //     time += dt;
    // }

    // GatherSolutions(u_loc, v_loc, h_loc, Nx_loc, Ny_loc, Nx, Ny, u, v, h, &local_mpi_info);

    // if (local_mpi_info.world_rank == root)
    // {

    //     // write to file
    //     ofstream vOut("output.txt", ios::out | ios ::trunc);
    //     if (vOut.is_open())
    //     {
    //         vOut.precision(5);
    //         for (int i = 0; i < Nx; ++i)
    //         {
    //             for (int j = 0; j < Ny; ++j)
    //             {
    //                 vOut << setw(15) << i * dx << setw(15) << j * dy << setw(15) << u[i * Ny + j] << setw(15) << v[i * Ny + j] << setw(15) << h[i * Ny + j]
    //                      << endl;
    //             }
    //         }
    //         vOut.close();
    //     }
    //     else
    //     {
    //         cout << "Unable to write to file" << endl;
    //     }
    // }

    // delete[] u;
    // delete[] v;
    // delete[] h;
    // delete[] u_loc;
    // delete[] v_loc;
    // delete[] h_loc;
    // delete[] h0_loc;

    // // debug output
    // cout << "====== h ======" << endl;
    // printMatrix(Nx,Ny,h);

    // // ======================================================
    // // test for evaluating f
    // double *fu_local = new double[Nx_local * Ny_local];
    // double *fv_local = new double[Nx_local * Ny_local];
    // double *fh_local = new double[Nx_local * Ny_local];
    // // // Evaluate_fu_BLAS(u, v, h, Nx, Ny, dx, dy, fu);
    // // // Evaluate_fv_BLAS(u, v, h, Nx, Ny, dx, dy, fv);
    // // // Evaluate_fh_BLAS(u, v, h, Nx, Ny, dx, dy, fh);

    // ======================================================
    // Write to file in root rank
    // Write initial condition
    // if (world_rank == root)
    // {
    //     ofstream vOut("output.txt", ios::out | ios ::trunc);
    //     vOut.precision(5);
    //     for (int j = 0; j < Nx; ++j)
    //     {
    //         for (int i = 0; i < Ny; ++i)
    //         {
    //             vOut << setw(15) << i * dx << setw(15) << j * dy << setw(15) << u[i * Nx + j] << setw(15) << v[i * Nx + j] << setw(15) << h[i * Nx + j]
    //                  << endl;
    //         }
    //     }
    // }

    // // deallocations
    // delete[] u;
    // delete[] v;
    // delete[] h;
    // delete[] h0;
    // delete[] fu;
    // delete[] fv;
    // delete[] fh;

    MPI_Finalize();
    return 0;
}
