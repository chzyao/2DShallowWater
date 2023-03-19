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

typedef struct
{
    // Store MPI info for each process
    MPI_Comm comm;
    int world_rank;
    int world_size;

    int dims[2];
    int coords[2];
    int west_rank;
    int east_rank;
    int north_rank;
    int south_rank;

} Local_MPI_Info;

void StoreMPIInfo(Local_MPI_Info *local_mpi_info, MPI_Comm comm_cartesian, const int *dims)
{
    const int ndims = 2;
    int periods[ndims] = {1, 1};
    // *local_mpi_info: pointer to the struct Local_MPI_Info
    local_mpi_info->comm = comm_cartesian;
    MPI_Comm_size(comm_cartesian, &(local_mpi_info->world_size));
    MPI_Comm_rank(comm_cartesian, &(local_mpi_info->world_rank));

    // dims info
    local_mpi_info->dims[0] = dims[0];
    local_mpi_info->dims[1] = dims[1];

    // Retrieve virtual topology info
    MPI_Cart_get(comm_cartesian, ndims, local_mpi_info->dims, periods,
                 local_mpi_info->coords);

    // identify neighbouring processes
    MPI_Cart_shift(local_mpi_info->comm, 0, 1, &(local_mpi_info->west_rank),
                   &(local_mpi_info->east_rank));
    MPI_Cart_shift(local_mpi_info->comm, 1, 1, &(local_mpi_info->north_rank),
                   &(local_mpi_info->south_rank));
}

void SetInitialConditions(double *u_loc, double *v_loc, double *h_loc,
                          double *h0_loc, int Nx_loc, int Ny_loc, int ic,
                          double dx, double dy,
                          Local_MPI_Info *local_mpi_info)
{
    int N_ghosts = 3; // ghost node on each side

    // Starting position coords (globally) of the subdomain (exclusing ghost
    // cells) we have 3 ghost cells each side
    int x_global = local_mpi_info->coords[0] * (Nx_loc - 2 * N_ghosts);
    int y_global = local_mpi_info->coords[1] * (Ny_loc - 2 * N_ghosts);

    for (int i = N_ghosts; i < Nx_loc - N_ghosts; ++i)
    {
        for (int j = N_ghosts; j < Ny_loc - N_ghosts; ++j)
        {
            int i_global = x_global + i;
            int j_global = y_global + j;
            u_loc[i * Ny_loc + j] = 0.0;
            v_loc[i * Ny_loc + j] = 0.0;

            if (ic == 1)
            {
                h0_loc[i * Ny_loc + j] =
                    10.0 + exp(-(i_global * dx - 50) * (i_global * dx - 50) / 25.0);
            }
            else if (ic == 2)
            {
                h0_loc[i * Ny_loc + j] =
                    10.0 + exp(-(j_global * dy - 50) * (j_global * dy - 50) / 25.0);
            }
            else if (ic == 3)
            {
                h0_loc[i * Ny_loc + j] =
                    10.0 + exp(-((i_global * dx - 50) * (i_global * dx - 50) +
                                 (j_global * dy - 50) * (j_global * dy - 50)) /
                               25.0);
            }
            else
            {
                h0_loc[i * Ny_loc + j] =
                    10.0 +
                    exp(-((i_global * dx - 25) * (i_global * dx - 25) +
                          (j_global * dy - 25) * (j_global * dy - 25)) /
                        25.0) +
                    exp(-((i_global * dx - 75) * (i_global * dx - 75) +
                          (j_global * dy - 75) * (j_global * dy - 75)) /
                        25.0);
            }
        }
    }

    // copy the initial surface height h0 to h as initial conditions
    cblas_dcopy(Nx_loc * Ny_loc, h0_loc, 1, h_loc, 1);
}

void GatherSolutions(double *u_loc, double *v_loc, double *h_loc, int Nx_loc,
                     int Ny_loc, int Nx, int Ny,
                     double *u, double *v, double *h,
                     Local_MPI_Info *local_mpi_info)
{
    // Gather in root process
    const int root = 0;

    int N_ghosts = 3; // ghost node on each side

    int local_grid_size = (Nx_loc - 2 * N_ghosts) * (Ny_loc - 2 * N_ghosts); // without ghost nodes introduced

    // column of subarray without ghost cells
    // stride: Nx_loc
    MPI_Datatype no_ghost_column_type;
    MPI_Type_vector(Ny_loc - 2 * N_ghosts, 1, Nx_loc, MPI_DOUBLE, &no_ghost_column_type);
    MPI_Type_commit(&no_ghost_column_type);

    int *recvcounts = nullptr;
    int *displs = nullptr;

    if (local_mpi_info->world_rank == root)
    {
        u = new double[Nx * Ny];
        v = new double[Nx * Ny];
        h = new double[Nx * Ny];
        recvcounts = new int[local_mpi_info->world_size]; // each entry: number of elements (local grid size) received from each process
        displs = new int[local_mpi_info->world_size];     //  starting index in received buffer from each process
    }
    for (int i = 0; i < local_mpi_info->world_size; ++i)
    {
        recvcounts[i] = local_grid_size;
        displs[i] = i * local_grid_size;
    }

    for (int j = 0; j < Nx_loc - 2 * N_ghosts; ++j)
    {
        int send_column_idx = N_ghosts + j * Ny_loc;

        // the receive columns should relate to the global domain
        int i_global = local_mpi_info->coords[0] * (Nx_loc - 2 * N_ghosts);
        int j_global = local_mpi_info->coords[1] * (Ny_loc - 2 * N_ghosts) + j;
        int recv_column_idx = i_global * Ny + j_global; // relate to the global domain

        std::cout << "Rank: " << local_mpi_info->world_rank
                  << " Send column index: " << send_column_idx
                  << " Receive column index: " << recv_column_idx << std::endl;

        // Print values before the MPI_Gatherv calls
        std::cout << "u_loc value: " << u_loc[send_column_idx] << std::endl;
        std::cout << "v_loc value: " << v_loc[send_column_idx] << std::endl;
        std::cout << "h_loc value: " << h_loc[send_column_idx] << std::endl;

        MPI_Gatherv(u_loc + send_column_idx, 1, no_ghost_column_type, u, recvcounts, displs, no_ghost_column_type, root, local_mpi_info->comm);
        MPI_Gatherv(v_loc + send_column_idx, 1, no_ghost_column_type, v, recvcounts, displs, no_ghost_column_type, root, local_mpi_info->comm);
        MPI_Gatherv(h_loc + send_column_idx, 1, no_ghost_column_type, h, recvcounts, displs, no_ghost_column_type, root, local_mpi_info->comm);
    }

    // Free the new MPI datatype
    MPI_Type_free(&no_ghost_column_type);

    if (local_mpi_info->world_rank == root)
    {
        delete[] recvcounts;
        delete[] displs;
    }
}

int main(int argc, char *argv[])
{
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
    // Cartesian Topology
    const int ndims = 2;
    int dims[ndims] = {0, 0};
    int periods[ndims] = {1, 1};
    MPI_Comm comm_cartesian;
    MPI_Dims_create(world_size, ndims, dims); // automatic division in grid
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm_cartesian);

    // Store MPI Information
    Local_MPI_Info local_mpi_info;
    StoreMPIInfo(&local_mpi_info, comm_cartesian, dims);

    // Subdomain ===============================================
    int Nx_loc = Nx / dims[0] + 6; // 3 ghost cells each side
    int Ny_loc = Ny / dims[1] + 6; // 3 ghost cells each side

    std::cout << "Nx_loc" << Nx_loc << std::endl;
    std::cout << "Ny_loc" << Ny_loc << std::endl;

    // Allocate solution memories (allocated with ghost cells)
    double *u_loc = new double[Nx_loc * Ny_loc];
    double *v_loc = new double[Nx_loc * Ny_loc];
    double *h_loc = new double[Nx_loc * Ny_loc];
    double *h0_loc = new double[Nx_loc * Ny_loc];

    // solutions
    double *u = nullptr;
    double *v = nullptr;
    double *h = nullptr;

    // =========================================================
    // test for SetInitialConditions
    SetInitialConditions(u_loc, v_loc, h_loc, h0_loc, Nx_loc, Ny_loc,
                         ic, dx, dy, &local_mpi_info);

    GatherSolutions(u_loc, v_loc, h_loc, Nx_loc, Ny_loc, Nx, Ny, u, v, h, &local_mpi_info);

    if (local_mpi_info.world_rank == root)
    {
        // write to file
        ofstream vOut("output.txt", ios::out | ios ::trunc);
        if (vOut.is_open())
        {
            vOut.precision(5);
            for (int i = 0; i < Nx; ++i)
            {
                for (int j = 0; j < Ny; ++j)
                {
                    vOut << setw(15) << i * dx << setw(15) << j * dy << setw(15) << u[i * Ny + j] << setw(15) << v[i * Ny + j] << setw(15) << h[i * Ny + j]
                         << endl;
                }
            }
            vOut.close();
        }
        else
        {
            cout << "Unable to write to file" << endl;
        }
    }

    delete[] u;
    delete[] v;
    delete[] h;
    delete[] u_loc;
    delete[] v_loc;
    delete[] h_loc;
    delete[] h0_loc;

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

    // // ======================================================
    // // 4th order RK Time Integrations

    // // Time advancement
    // double time = 0.0; // start time
    // while (time <= T)
    // {
    //     TimeIntegration(u_local, v_local, h_local, Nx_local, Ny_local, Nx, Ny,
    //     dx, dy, dt, fu_local, fv_local, fh_local, &local_mpi_info); time += dt;
    // }

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

    MPI_Comm_free(&comm_cartesian);
    MPI_Finalize();
    return 0;
}
