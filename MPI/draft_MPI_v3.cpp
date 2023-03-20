#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
// #include <mpi.h>
#include <cstdlib>
#include <cblas.h>


using namespace std;
namespace po = boost::program_options;

void SetInitialConditions(double *u, double *v, double *h, double *h0, int Nx,
                          int Ny, int ic, double dx, double dy)
{
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            u[i * Ny + j] = 0.0;
            v[i * Ny + j] = 0.0;
            if (ic == 1)
            {
                h0[i * Ny + j] = 10.0 + exp(-(i * dx - 50) * (i * dx - 50) / 25.0);
            }
            else if (ic == 2)
            {
                h0[i * Ny + j] = 10.0 + exp(-(j * dy - 50) * (j * dy - 50) / 25.0);
            }
            else if (ic == 3)
            {
                h0[i * Ny + j] = 10.0 + exp(
                                            -((i * dx - 50) * (i * dx - 50) + (j * dy - 50) * (j * dy - 50)) /
                                            25.0);
            }
            else
            {
                h0[i * Ny + j] = 10.0 + exp(-((i * dx - 25) * (i * dx - 25) + (j * dy - 25) * (j * dy - 25)) / 25.0) +
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

        for (int i = 0; i < Nx; ++i)
        {

            for (int j = 0; j < Ny; ++j)
            {
                deriv[i * Ny + j] =
                    px *
                    (-u[((i - 3 + Nx) % Nx) * Ny + j] / 60.0 + 3.0 / 20.0 * u[((i - 2 + Nx) % Nx) * Ny + j] -
                     3.0 / 4.0 * u[((i - 1 + Nx) % Nx) * Ny + j] + 3.0 / 4.0 * u[((i + 1) % Nx) * Ny + j] -
                     3.0 / 20.0 * u[((i + 2) % Nx) * Ny + j] + u[((i + 3) % Nx) * Ny + j] / 60.0);
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
                deriv[i * Ny + j] =
                    py *
                    (-u[i * Ny + (j - 3 + Ny) % Ny] / 60.0 + 3.0 / 20.0 * u[i * Ny + (j - 2 + Ny) % Ny] -
                     3.0 / 4.0 * u[i * Ny + (j - 1 + Ny) % Ny] + 3.0 / 4.0 * u[i * Ny + (j + 1) % Ny] -
                     3.0 / 20.0 * u[i * Ny + (j + 2) % Ny] + u[i * Ny + (j + 3) % Ny] / 60.0);
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

    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            f[i * Ny + j] = -u[i * Ny + j] * deriux[i * Ny + j] -
                            v[i * Ny + j] * deriuy[i * Ny + j] -
                            g * derihx[i * Ny + j];
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

    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            f[i * Ny + j] = -u[i * Ny + j] * derivx[i * Ny + j] -
                            v[i * Ny + j] * derivy[i * Ny + j] -
                            g * derihy[i * Ny + j];
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
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            hu[i * Ny + j] = h[i * Ny + j] * u[i * Ny + j];
            hv[i * Ny + j] = h[i * Ny + j] * v[i * Ny + j];
        }
    }

    SpatialDiscretisation(hu, Nx, Ny, dx, dy, 'x', derihux);
    SpatialDiscretisation(hv, Nx, Ny, dx, dy, 'y', derihvy);

    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            f[i * Ny + j] = -derihux[i * Ny + j] - derihvy[i * Ny + j];
        }
    }

    delete[] derihux;
    delete[] derihvy;
    delete[] hu;
    delete[] hv;
}

void TimeIntegration(double *u, double *v, double *h, int Nx, int Ny,
                     double dx, double dy, double dt, double *fu,
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
    double *tv = new double[Nx * Ny]; // temp vector t = v
    double *th = new double[Nx * Ny]; // temp vector t = h

    // Calculating k1 = f(yn) ===================================
    cblas_dcopy(Nx * Ny, u, 1, tu, 1);
    cblas_dcopy(Nx * Ny, v, 1, tv, 1);
    cblas_dcopy(Nx * Ny, h, 1, th, 1);

    Evaluate_fu(u, v, h, Nx, Ny, dx, dy, fu);
    Evaluate_fv(u, v, h, Nx, Ny, dx, dy, fv);
    Evaluate_fh(u, v, h, Nx, Ny, dx, dy, fh);

    // Evaluate_fu_BLAS(u, v, h, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(u, v, h, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(u, v, h, Nx, Ny, dx, dy, fh);

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
    Evaluate_fu(tu, tv, th, Nx, Ny, dx, dy, fu);
    Evaluate_fv(tu, tv, th, Nx, Ny, dx, dy, fv);
    Evaluate_fh(tu, tv, th, Nx, Ny, dx, dy, fh);

    // Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

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

    Evaluate_fu(tu, tv, th, Nx, Ny, dx, dy, fu);
    Evaluate_fv(tu, tv, th, Nx, Ny, dx, dy, fv);
    Evaluate_fh(tu, tv, th, Nx, Ny, dx, dy, fh);

    // Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

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

    Evaluate_fu(tu, tv, th, Nx, Ny, dx, dy, fu);
    Evaluate_fv(tu, tv, th, Nx, Ny, dx, dy, fv);
    Evaluate_fh(tu, tv, th, Nx, Ny, dx, dy, fh);

    // Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

    cblas_dcopy(Nx * Ny, fu, 1, k4_u, 1);
    cblas_dcopy(Nx * Ny, fv, 1, k4_v, 1);
    cblas_dcopy(Nx * Ny, fh, 1, k4_h, 1);

    // yn+1 = yn + 1/6*(k1+2*k2+2*k3+k4)*dt
    // Update solution
    for (int i = 0; i < Nx; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            u[i * Ny + j] += dt / 6.0 *
                             (k1_u[i * Ny + j] + 2.0 * k2_u[i * Ny + j] +
                              2.0 * k3_u[i * Ny + j] + k4_u[i * Ny + j]);
            v[i * Ny + j] += dt / 6.0 *
                             (k1_v[i * Ny + j] + 2.0 * k2_v[i * Ny + j] +
                              2.0 * k3_v[i * Ny + j] + k4_v[i * Ny + j]);
            h[i * Ny + j] += dt / 6.0 *
                             (k1_h[i * Ny + j] + 2.0 * k2_h[i * Ny + j] +
                              2.0 * k3_h[i * Ny + j] + k4_h[i * Ny + j]);
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

    // Read parameters from command line =========================

    po::options_description options("Available Options.");
    options.add_options()("help", "Display help message")(
        "dt", po::value<double>()->default_value(0.1), "Time-step to use")(
        "T", po::value<double>()->default_value(80.0), "Total integration time")(
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

    // Allocate solution memories
    double *u = new double[Nx * Ny];
    double *v = new double[Nx * Ny];
    double *h = new double[Nx * Ny];
    double *h0 = new double[Nx * Ny];

    // MPI =====================================================
    // MPI_Init(&argc, &argv);
    // const int root = 0; // root rank

    // // Get size and rank
    // int world_rank, world_size, retval_rank, retval_size;
    // retval_rank = MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // retval_size = MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // if (retval_rank == MPI_ERR_COMM || retval_size == MPI_ERR_COMM)
    // {
    //     std::cout << "Invalid communicator" << std::endl;
    //     return 1;
    // }

    // std::cout << "Goodbye World" << std::endl;

    // Subdomain ===============================================
    int N_ghosts = 3;
    // int Nx_loc = Nx / world_size; // 3 ghost cells each side
    // int Nx_loc_halo = Nx / world_size + 2 * N_ghosts;


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

    // ======================================================
    // test for SetInitialConditions
    SetInitialConditions(u, v, h, h0, Nx, Ny, ic, dx, dy);

    // // debug output
    // cout << "====== h ======" << endl;
    // printMatrix(Nx,Ny,h);

    // ======================================================
    // test for evaluating f
    double *fu = new double[Nx * Ny];
    double *fv = new double[Nx * Ny];
    double *fh = new double[Nx * Ny];

    // ======================================================
    // 4th order RK Time Integrations

    // Time advancement
    double time = 0.0; // start time
    while (time <= T)
    {
        TimeIntegration(u, v, h, Nx, Ny, dx, dy, dt, fu, fv, fh);
        time += dt;
    }

    ofstream vOut("output.txt", ios::out | ios ::trunc);
    vOut.precision(5);
    for (int j = 0; j < Nx; ++j)
    {
        for (int i = 0; i < Ny; ++i)
        {
            vOut << setw(15) << i * dx << setw(15) << j * dy << setw(15) << u[i * Nx + j] << setw(15) << v[i * Nx + j] << setw(15) << h[i * Nx + j] << endl;
        }
    }

    // deallocations
    delete[] u;
    delete[] v;
    delete[] h;
    delete[] h0;
    delete[] fu;
    delete[] fv;
    delete[] fh;

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

    return 0;
}
