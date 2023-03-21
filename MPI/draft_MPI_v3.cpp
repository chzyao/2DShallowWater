#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <cblas.h>

using namespace std;
namespace po = boost::program_options;

typedef struct
{
    // Store MPI info for each process
    MPI_Comm comm;
    int world_rank;
    int world_size;

    int dims[1];
    int coords[1];
    int west_rank;
    int east_rank;

} Local_MPI_Info;

void StoreMPIInfo(Local_MPI_Info *local_mpi_info, MPI_Comm comm_cartesian, const int *dims)
{
    const int ndims = 1;
    int periods[ndims] = {1};
    // *local_mpi_info: pointer to the struct Local_MPI_Info
    local_mpi_info->comm = comm_cartesian;
    MPI_Comm_size(comm_cartesian, &(local_mpi_info->world_size));
    MPI_Comm_rank(comm_cartesian, &(local_mpi_info->world_rank));

    // dims info
    local_mpi_info->dims[0] = dims[0];

    // Retrieve virtual topology info
    MPI_Cart_get(comm_cartesian, ndims, local_mpi_info->dims, periods,
                 local_mpi_info->coords);

    // identify neighbouring processes
    MPI_Cart_shift(local_mpi_info->comm, 0, 1, &(local_mpi_info->west_rank),
                   &(local_mpi_info->east_rank));
}

void LocalBCInfoExchange(double *u_loc, int Nx_loc, int Ny, Local_MPI_Info *local_mpi_info)
{
    // Find the positions in the global domain for this current rank
    const int N_ghosts = 3;
    int x_global = local_mpi_info->coords[0] * (Nx_loc - 2 * N_ghosts);

    // buffers for sending and receiving data
    // x-dir in cartesian grid
    double *send_west = new double[Ny * N_ghosts];
    double *recv_west = new double[Ny * N_ghosts];
    double *send_east = new double[Ny * N_ghosts];
    double *recv_east = new double[Ny * N_ghosts];

    // Boundary info exchange in x-dir =====================================
    // send buffer x-dir
    for (int j = 0; j < Ny; ++j)
    {
        for (int k = 0; k < N_ghosts; ++k)
        {

            send_west[k * Ny + j] = u_loc[(N_ghosts + k) * Ny + j];
            // cout << "Send West " << send_west[k * Ny + j] << " " << (N_ghosts + k) << " " << j << endl;
            send_east[k * Ny + j] = u_loc[(Nx_loc - 2 * N_ghosts + k) * Ny + j];
            // cout << "Send East " << send_east[k * Ny + j] << " " << (Nx_loc - 2 * N_ghosts + k) << " " << j << endl;
        }
    }

    // Exchange boundary info
    MPI_Sendrecv(send_west, Ny * N_ghosts, MPI_DOUBLE, local_mpi_info->west_rank, 0, recv_east, Ny * N_ghosts, MPI_DOUBLE, local_mpi_info->east_rank, 0, local_mpi_info->comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_east, Ny * N_ghosts, MPI_DOUBLE, local_mpi_info->east_rank, 1, recv_west, Ny * N_ghosts, MPI_DOUBLE, local_mpi_info->west_rank, 1, local_mpi_info->comm, MPI_STATUS_IGNORE);

    // Update boundary info
    // x-dir
    for (int j = 0; j < Ny; ++j)
    {
        for (int k = 0; k < N_ghosts; ++k)
        {
            u_loc[(Nx_loc - N_ghosts + k) * Ny + j] = recv_east[k * Ny + j];
            // cout << "recv from east " << recv_east[k * Ny + j] << "  " << (Nx_loc - N_ghosts + k) << " " << j << endl;
            u_loc[k * Ny + j] = recv_west[k * Ny + j];
            // cout << "recv from west " << recv_west[k * Ny + j] << "  " << k << " " << j << endl;
        }
    }

    // Deallocation
    delete[] send_west;
    delete[] send_east;

    delete[] recv_west;
    delete[] recv_east;
}

void SetInitialConditions(double *u, double *v, double *h, double *h0, int Nx,
                          int Ny, int Nx_loc, int ic, double dx, double dy, Local_MPI_Info *local_mpi_info)
{
    // Find the positions in the global domain for this current rank
    const int N_ghosts = 3;
    int x_global = local_mpi_info->coords[0] * (Nx_loc - 2 * N_ghosts);

    for (int i = N_ghosts; i < Nx_loc - N_ghosts; ++i)
    {
        int i_global = x_global + i;
        for (int j = 0; j < Ny; ++j)
        {
            u[i * Ny + j] = 0.0;
            v[i * Ny + j] = 0.0;
            if (ic == 1)
            {
                h0[i * Ny + j] = 10.0 + exp(-(i_global * dx - 50) * (i_global * dx - 50) / 25.0);
            }
            else if (ic == 2)
            {
                h0[i * Ny + j] = 10.0 + exp(-(j * dy - 50) * (j * dy - 50) / 25.0);
            }
            else if (ic == 3)
            {
                h0[i * Ny + j] = 10.0 + exp(
                                            -((i_global * dx - 50) * (i_global * dx - 50) + (j * dy - 50) * (j * dy - 50)) /
                                            25.0);
            }
            else
            {
                h0[i * Ny + j] = 10.0 + exp(-((i_global * dx - 25) * (i_global * dx - 25) + (j * dy - 25) * (j * dy - 25)) / 25.0) +
                                 exp(-((i_global * dx - 75) * (i_global * dx - 75) + (j * dy - 75) * (j * dy - 75)) / 25.0);
            }
        }
    }

    // copy the initial surface height h0 to h as initial conditions
    cblas_dcopy(Nx * Ny, h0, 1, h, 1);
}

void SpatialDiscretisation(double *u, int Nx, int Ny, int Nx_loc, double dx, double dy, char dir, double *deriv_loc, Local_MPI_Info *local_mpi_info)
{
    // Find the positions in the global domain for this current rank
    const int N_ghosts = 3;
    int x_global = local_mpi_info->coords[0] * (Nx_loc - 2 * N_ghosts);
    // Discretisation in x-dir ============================================
    if (dir == 'x')
    {
        double px = 1.0 / dx;

        for (int i = N_ghosts; i < Nx_loc - N_ghosts; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                deriv_loc[i * Ny + j] =
                    px *
                    (-u[(i - 3) * Ny + j] / 60.0 + 3.0 / 20.0 * u[(i - 2) * Ny + j] -
                     3.0 / 4.0 * u[(i - 1) * Ny + j] + 3.0 / 4.0 * u[(i + 1) * Ny + j] -
                     3.0 / 20.0 * u[(i + 2) * Ny + j] + u[(i + 3) * Ny + j] / 60.0);
            }
        }
    }

    // Discretisation in y-dir ============================================
    else if (dir == 'y')
    {
        double py = 1.0 / dy;

        for (int i = N_ghosts; i < Nx_loc - N_ghosts; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                deriv_loc[i * Ny + j] =
                    py *
                    (-u[i * Ny + (j - 3)] / 60.0 + 3.0 / 20.0 * u[i * Ny + (j - 2)] -
                     3.0 / 4.0 * u[i * Ny + (j - 1)] + 3.0 / 4.0 * u[i * Ny + (j + 1)] -
                     3.0 / 20.0 * u[i * Ny + (j + 2)] + u[i * Ny + (j + 3)] / 60.0);
            }
        }
    }

    LocalBCInfoExchange(deriv_loc, Nx_loc, Ny, local_mpi_info);
}

void Evaluate_fu(double *u, double *v, double *h, int Nx, int Nx_loc, int Ny,
                 double dx, double dy, double *f_loc, Local_MPI_Info *local_mpi_info)
{
    const int N_ghosts = 3;
    double g = 9.81;
    double *deriux = new double[Nx_loc * Ny];
    double *deriuy = new double[Nx_loc * Ny];
    double *derihx = new double[Nx_loc * Ny];

    SpatialDiscretisation(u, Nx, Ny, Nx_loc, dx, dy, 'x', deriux, local_mpi_info);
    SpatialDiscretisation(u, Nx, Ny, Nx_loc, dx, dy, 'y', deriuy, local_mpi_info);
    SpatialDiscretisation(h, Nx, Ny, Nx_loc, dx, dy, 'x', derihx, local_mpi_info);

    for (int i = N_ghosts; i < Nx_loc - N_ghosts; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            f_loc[i * Ny + j] = -u[i * Ny + j] * deriux[i * Ny + j] - v[i * Ny + j] * deriuy[i * Ny + j] - g * derihx[i * Ny + j];
        }
    }

    delete[] deriux;
    delete[] deriuy;
    delete[] derihx;
}

void Evaluate_fv(double *u, double *v, double *h, int Nx, int Nx_loc, int Ny,
                 double dx, double dy, double *f_loc, Local_MPI_Info *local_mpi_info)
{
    double g = 9.81;
    const int N_ghosts = 3;

    double *derivx = new double[Nx_loc * Ny];
    double *derivy = new double[Nx_loc * Ny];
    double *derihy = new double[Nx_loc * Ny];

    SpatialDiscretisation(v, Nx, Ny, Nx_loc, dx, dy, 'x', derivx, local_mpi_info);
    SpatialDiscretisation(v, Nx, Ny, Nx_loc, dx, dy, 'y', derivy, local_mpi_info);
    SpatialDiscretisation(h, Nx, Ny, Nx_loc, dx, dy, 'y', derihy, local_mpi_info);

    for (int i = N_ghosts; i < Nx_loc - N_ghosts; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            f_loc[i * Ny + j] = -u[i * Ny + j] * derivx[i * Ny + j] - v[i * Ny + j] * derivy[i * Ny + j] - g * derihy[i * Ny + j];
        }
    }

    delete[] derivx;
    delete[] derivy;
    delete[] derihy;
}

void Evaluate_fh(double *u, double *v, double *h, int Nx, int Nx_loc, int Ny,
                 double dx, double dy, double *f_loc, Local_MPI_Info *local_mpi_info)
{
    double *deriux = new double[Nx_loc * Ny];
    double *derihx = new double[Nx_loc * Ny];
    double *derivy = new double[Nx_loc * Ny];
    double *derihy = new double[Nx_loc * Ny];

    const int N_ghosts = 3;

    SpatialDiscretisation(u, Nx, Ny, Nx_loc, dx, dy, 'x', deriux, local_mpi_info);
    SpatialDiscretisation(h, Nx, Ny, Nx_loc, dx, dy, 'x', derihx, local_mpi_info);
    SpatialDiscretisation(v, Nx, Ny, Nx_loc, dx, dy, 'y', derivy, local_mpi_info);
    SpatialDiscretisation(h, Nx, Ny, Nx_loc, dx, dy, 'y', derihy, local_mpi_info);

    for (int i = N_ghosts; i < Nx_loc - N_ghosts; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            f_loc[i * Ny + j] = -h[i * Ny + j] * deriux[i * Ny + j] - u[i * Ny + j] * derihx[i * Ny + j] - h[i * Ny + j] * derivy[i * Ny + j] - v[i * Ny + j] * derihy[i * Ny + j];
        }
    }

    delete[] deriux;
    delete[] derihx;
    delete[] derivy;
    delete[] derihy;
}

void TimeIntegration(double *u, double *v, double *h, int Nx, int Nx_loc,
                     int Ny, double dx, double dy, double dt, double *fu,
                     double *fv, double *fh, double *u_loc, double *v_loc, double *h_loc, Local_MPI_Info *local_mpi_info)
{
    // Solving for u
    double *k1_u = new double[Nx_loc * Ny];
    double *k2_u = new double[Nx_loc * Ny];
    double *k3_u = new double[Nx_loc * Ny];
    double *k4_u = new double[Nx_loc * Ny];

    // Solve for v
    double *k1_v = new double[Nx_loc * Ny];
    double *k2_v = new double[Nx_loc * Ny];
    double *k3_v = new double[Nx_loc * Ny];
    double *k4_v = new double[Nx_loc * Ny];

    // Solve for h
    double *k1_h = new double[Nx_loc * Ny];
    double *k2_h = new double[Nx_loc * Ny];
    double *k3_h = new double[Nx_loc * Ny];
    double *k4_h = new double[Nx_loc * Ny];

    double *tu = new double[Nx_loc * Ny]; // temp vector t = u
    double *tv = new double[Nx_loc * Ny]; // temp vector t = v
    double *th = new double[Nx_loc * Ny]; // temp vector t = h

    // Calculating k1 = f(yn) ===================================
    cblas_dcopy(Nx_loc * Ny, u, 1, tu, 1);
    cblas_dcopy(Nx_loc * Ny, v, 1, tv, 1);
    cblas_dcopy(Nx_loc * Ny, h, 1, th, 1);

    Evaluate_fu(u, v, h, Nx, Nx_loc, Ny, dx, dy, fu, local_mpi_info);
    Evaluate_fv(u, v, h, Nx, Nx_loc, Ny, dx, dy, fv, local_mpi_info);
    Evaluate_fh(u, v, h, Nx, Nx_loc, Ny, dx, dy, fh, local_mpi_info);

    // Evaluate_fu_BLAS(u, v, h, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(u, v, h, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(u, v, h, Nx, Ny, dx, dy, fh);

    cblas_dcopy(Nx_loc * Ny, fu, 1, k1_u, 1);
    cblas_dcopy(Nx_loc * Ny, fv, 1, k1_v, 1);
    cblas_dcopy(Nx_loc * Ny, fh, 1, k1_h, 1);

    // Calculating k2 = f(yn + dt*k1/2) ==========================
    // reset temp values
    cblas_dcopy(Nx_loc * Ny, u, 1, tu, 1); // reset tu to u
    cblas_dcopy(Nx_loc * Ny, v, 1, tv, 1);
    cblas_dcopy(Nx_loc * Ny, h, 1, th, 1);

    // update un to un+dt*k1/2 to evaluate f for k2
    cblas_daxpy(Nx_loc * Ny, dt / 2.0, k1_u, 1, tu, 1);
    cblas_daxpy(Nx_loc * Ny, dt / 2.0, k1_v, 1, tv, 1);
    cblas_daxpy(Nx_loc * Ny, dt / 2.0, k1_h, 1, th, 1);

    // Evaluate new f
    Evaluate_fu(tu, tv, th, Nx, Nx_loc, Ny, dx, dy, fu, local_mpi_info);
    Evaluate_fv(tu, tv, th, Nx, Nx_loc, Ny, dx, dy, fv, local_mpi_info);
    Evaluate_fh(tu, tv, th, Nx, Nx_loc, Ny, dx, dy, fh, local_mpi_info);

    // Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

    cblas_dcopy(Nx_loc * Ny, fu, 1, k2_u, 1);
    cblas_dcopy(Nx_loc * Ny, fv, 1, k2_v, 1);
    cblas_dcopy(Nx_loc * Ny, fh, 1, k2_h, 1);

    // Calculating k3 = f(yn+dt*k2/2) =============================
    // reset temp values
    cblas_dcopy(Nx_loc * Ny, u, 1, tu, 1); // reset tu to u
    cblas_dcopy(Nx_loc * Ny, v, 1, tv, 1);
    cblas_dcopy(Nx_loc * Ny, h, 1, th, 1);

    // update un to un+dt*k2/2 to evaluate f for k3
    cblas_daxpy(Nx_loc * Ny, dt / 2.0, k2_u, 1, tu, 1);
    cblas_daxpy(Nx_loc * Ny, dt / 2.0, k2_v, 1, tv, 1);
    cblas_daxpy(Nx_loc * Ny, dt / 2.0, k2_h, 1, th, 1);

    Evaluate_fu(tu, tv, th, Nx, Nx_loc, Ny, dx, dy, fu, local_mpi_info);
    Evaluate_fv(tu, tv, th, Nx, Nx_loc, Ny, dx, dy, fv, local_mpi_info);
    Evaluate_fh(tu, tv, th, Nx, Nx_loc, Ny, dx, dy, fh, local_mpi_info);

    // Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

    cblas_dcopy(Nx_loc * Ny, fu, 1, k3_u, 1);
    cblas_dcopy(Nx_loc * Ny, fv, 1, k3_v, 1);
    cblas_dcopy(Nx_loc * Ny, fh, 1, k3_h, 1);

    // k4 = f(yn+dt*k3) ===========================================
    // reset temp values
    cblas_dcopy(Nx_loc * Ny, u, 1, tu, 1); // reset tu to u
    cblas_dcopy(Nx_loc * Ny, v, 1, tv, 1);
    cblas_dcopy(Nx_loc * Ny, h, 1, th, 1);

    // update un to un+dt*k2/2 to evaluate f for k3
    cblas_daxpy(Nx_loc * Ny, dt, k3_u, 1, tu, 1);
    cblas_daxpy(Nx_loc * Ny, dt, k3_v, 1, tv, 1);
    cblas_daxpy(Nx_loc * Ny, dt, k3_h, 1, th, 1);

    Evaluate_fu(tu, tv, th, Nx, Nx_loc, Ny, dx, dy, fu, local_mpi_info);
    Evaluate_fv(tu, tv, th, Nx, Nx_loc, Ny, dx, dy, fv, local_mpi_info);
    Evaluate_fh(tu, tv, th, Nx, Nx_loc, Ny, dx, dy, fh, local_mpi_info);

    // Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

    cblas_dcopy(Nx_loc * Ny, fu, 1, k4_u, 1);
    cblas_dcopy(Nx_loc * Ny, fv, 1, k4_v, 1);
    cblas_dcopy(Nx_loc * Ny, fh, 1, k4_h, 1);

    // yn+1 = yn + 1/6*(k1+2*k2+2*k3+k4)*dt
    // Update solution
    const int N_ghosts = 3;
    int x_global = local_mpi_info->coords[0] * (Nx_loc - 2 * N_ghosts);

    for (int i = N_ghosts; i < Nx_loc - N_ghosts; ++i)
    {
        for (int j = 0; j < Ny; ++j)
        {
            u_loc[i * Ny + j] += dt / 6.0 *
                                 (k1_u[i * Ny + j] + 2.0 * k2_u[i * Ny + j] +
                                  2.0 * k3_u[i * Ny + j] + k4_u[i * Ny + j]);
            v_loc[i * Ny + j] += dt / 6.0 *
                                 (k1_v[i * Ny + j] + 2.0 * k2_v[i * Ny + j] +
                                  2.0 * k3_v[i * Ny + j] + k4_v[i * Ny + j]);
            h_loc[i * Ny + j] += dt / 6.0 *
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

void CollectSolutions(double *u_loc, double *v_loc, double *h_loc, int Nx_loc, double *u, double *v, double *h, int Nx, int Ny, double dx, double dy, Local_MPI_Info *local_mpi_info)
{
    cout << "Collecting solutions" << endl;
    const int N_ghosts = 3;
    const int root = 0;
    int *recvcounts = new int[local_mpi_info->world_size];
    int *displs = new int[local_mpi_info->world_size];

    int local_size = (Nx_loc - 2 * N_ghosts) * Ny;
    for (int i = 0; i < local_mpi_info->world_size; ++i)
    {
        recvcounts[i] = local_size;
        displs[i] = i * local_size;
    }
    if (local_mpi_info->world_rank == 0) // Allocate memory only for the root process
    {
        u = new double[Nx * Ny];
        v = new double[Nx * Ny];
        h = new double[Nx * Ny];
    }
    MPI_Gatherv(u_loc, local_size, MPI_DOUBLE, u, recvcounts, displs, MPI_DOUBLE, root, local_mpi_info->comm);
    MPI_Gatherv(v_loc, local_size, MPI_DOUBLE, v, recvcounts, displs, MPI_DOUBLE, root, local_mpi_info->comm);
    MPI_Gatherv(h_loc, local_size, MPI_DOUBLE, h, recvcounts, displs, MPI_DOUBLE, root, local_mpi_info->comm);

    if (local_mpi_info->world_rank == root)
    {
        // Write to file

        ofstream vOut("output.txt", ios::out | ios ::trunc);
        vOut.precision(5);
        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < Ny; ++j)
            {
                vOut << setw(15) << i * dx << setw(15) << j * dy << setw(15) << u[i * Ny + j] << setw(15) << v[i * Ny + j] << setw(15) << h[i * Ny + j] << endl;
            }
        }
    }
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
    // Cartesian Topology
    const int ndims = 1;
    int dims[ndims] = {0};
    int periods[ndims] = {1};
    MPI_Comm comm_cartesian;
    MPI_Dims_create(world_size, ndims, dims); // automatic division in grid
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm_cartesian);

    // Store MPI Information
    Local_MPI_Info local_mpi_info;
    StoreMPIInfo(&local_mpi_info, comm_cartesian, dims);

    int Nx_loc = Nx / dims[0] + 6; // 3 ghost cells each side
    cout << Nx_loc << endl;

    // std::cout << "Nx_loc" << Nx_loc << std::endl;
    // std::cout << "Ny_loc" << Ny_loc << std::endl;

    // // Allocate solution memories (allocated with ghost cells)
    // double *u_loc = new double[Nx_loc * Ny_loc];
    // double *v_loc = new double[Nx_loc * Ny_loc];
    // double *h_loc = new double[Nx_loc * Ny_loc];
    // double *h0_loc = new double[Nx_loc * Ny_loc];
    // Allocate solution memories
    double *u = new double[Nx_loc * Ny];
    double *v = new double[Nx_loc * Ny];
    double *h = new double[Nx_loc * Ny];
    double *h0 = new double[Nx_loc * Ny];

    double *fu_loc = new double[Nx_loc * Ny];
    double *fv_loc = new double[Nx_loc * Ny];
    double *fh_loc = new double[Nx_loc * Ny];

    double *u_loc = new double[Nx_loc * Ny];
    double *v_loc = new double[Nx_loc * Ny];
    double *h_loc = new double[Nx_loc * Ny];

    double *u_global = nullptr;
    double *v_global = nullptr;
    double *h_global = nullptr;

    // // Collecting results
    // if (world_rank == root)
    // {
    //     u_global = new double[Nx * Ny];
    //     v_global = new double[Nx * Ny];
    //     h_global = new double[Nx * Ny];
    // }

    // ======================================================
    // test for SetInitialConditions
    SetInitialConditions(u, v, h, h0, Nx, Ny, Nx_loc, ic, dx, dy, &local_mpi_info);

    // CollectSolutions(u, v, h, Nx_loc, u_global, v_global, h_global, Nx, Ny, dx, dy, &local_mpi_info);

    // // // debug output
    // // cout << "====== h ======" << endl;
    // // printMatrix(Nx,Ny,h);

    // ======================================================
    // test for evaluating f
    double *fu = new double[Nx_loc * Ny];
    double *fv = new double[Nx_loc * Ny];
    double *fh = new double[Nx_loc * Ny];

    // Evaluate_fu(u, v, h, Nx, Nx_loc, Ny, dx, dy, fu_loc, &local_mpi_info);

    // Evaluate_fv(u, v, h, Nx, Nx_loc, Ny, dx, dy, fv_loc, &local_mpi_info);

    // Evaluate_fh(u, v, h, Nx, Nx_loc, Ny, dx, dy, fh_loc, &local_mpi_info);
    // cout << "Fucked up" << endl;

    // ======================================================
    // 4th order RK Time Integrations
    // Time advancement
    double time = 0.0; // start time
    while (time <= T)
    {
        TimeIntegration(u, v, h, Nx, Nx_loc, Ny, dx, dy, dt, fu_loc, fv_loc, fh_loc, u_loc, v_loc, h_loc, &local_mpi_info);
        time += dt;
    }

    CollectSolutions(u_loc, v_loc, h_loc, Nx_loc, u_global, v_global, h_global, Nx, Ny, dx, dy, &local_mpi_info);

    // deallocations
    delete[] u;
    delete[] v;
    delete[] h;
    delete[] u_loc;
    delete[] v_loc;
    delete[] h_loc;
    delete[] fu_loc;
    delete[] fv_loc;
    delete[] fh_loc;

    // delete[] h0;
    // delete[] fu;
    // delete[] fv;
    // delete[] fh;

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

    // // deallocations
    // delete[] u;
    // delete[] v;
    // delete[] h;
    // delete[] h0;

    MPI_Comm_free(&comm_cartesian);
    MPI_Finalize();
    return 0;
}
