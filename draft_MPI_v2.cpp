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

void SpatialDiscretisation(double *u, int Nx_loc, int Ny_loc,
                           double dx, double dy,
                           char dir, double *deriv_loc)
{
    const int N_ghosts = 3;
    // Discretisation in x-dir ============================================
    if (dir == 'x')
    {
        double px = 1.0 / dx;

        for (int i = N_ghosts; i < Nx_loc-N_ghosts; ++i)
        {
            for (int j = 0; j < Ny_loc; ++j)
            {
                deriv_loc[i * Ny_loc + j] =
                    px *
                    (-u[(i - 3) * Ny_loc + j] / 60.0 + 3.0 / 20.0 * u[(i - 2) * Ny_loc + j] -
                     3.0 / 4.0 * u[(i - 1) * Ny_loc + j] + 3.0 / 4.0 * u[(i + 1) * Ny_loc + j] -
                     3.0 / 20.0 * u[(i + 2) * Ny_loc + j] + u[(i + 3) * Ny_loc + j] / 60.0);
            }
        }
    }

    // Discretisation in y-dir ============================================
    else if (dir == 'y')
    {
        double py = 1.0 / dy;

        for (int i = 0; i < Nx_loc; ++i)
        {
            for (int j = N_ghosts; j < Ny_loc - N_ghosts; ++j)
            {
                deriv_loc[i * Ny_loc + j] =
                    py *
                    (-u[i * Ny_loc + (j - 3)] / 60.0 + 3.0 / 20.0 * u[i * Ny_loc + (j - 2 + Ny_loc)] -
                     3.0 / 4.0 * u[i * Ny_loc + (j - 1)] + 3.0 / 4.0 * u[i * Ny_loc + (j + 1)] -
                     3.0 / 20.0 * u[i * Ny_loc + (j + 2)] + u[i * Ny_loc + (j + 3)] / 60.0);
            }
        }
    }
}

void LocalBCInfoExchange(double *u_loc, int Nx_loc, int Ny_loc, Local_MPI_Info *local_mpi_info)
{
    // ghost cells on each side
    int N_ghosts = 3;

    // buffers for sending and receiving data
    // x-dir in cartesian grid
    double *send_west = new double[Ny_loc * N_ghosts];
    double *recv_west = new double[Ny_loc * N_ghosts];
    double *send_east = new double[Ny_loc * N_ghosts];
    double *recv_east = new double[Ny_loc * N_ghosts];

    // // y-dir in cartesian grid
    double *send_north = new double[Nx_loc * N_ghosts];
    double *recv_north = new double[Nx_loc * N_ghosts];
    double *send_south = new double[Nx_loc * N_ghosts];
    double *recv_south = new double[Nx_loc * N_ghosts];

    // Boundary info exchange in x-dir =====================================
    // send buffer x-dir
    for (int j = 0; j < Ny_loc; ++j)
    {
        for (int k = 0; k < N_ghosts; ++k)
        {
            send_west[k * Ny_loc + j] = u_loc[j * Nx_loc + N_ghosts + k];
            send_east[k * Ny_loc + j] = u_loc[j * Nx_loc + Nx_loc - 2 * N_ghosts + k];
        }
    }

    // Exchange boundary info
    MPI_Sendrecv(send_west, Ny_loc * N_ghosts, MPI_DOUBLE, local_mpi_info->west_rank, 0, recv_east, Ny_loc * N_ghosts, MPI_DOUBLE, local_mpi_info->east_rank, 0, local_mpi_info->comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_east, Ny_loc * N_ghosts, MPI_DOUBLE, local_mpi_info->east_rank, 1, recv_west, Ny_loc * N_ghosts, MPI_DOUBLE, local_mpi_info->west_rank, 1, local_mpi_info->comm, MPI_STATUS_IGNORE);

    // Update boundary info
    // x-dir
    for (int j = 0; j < Ny_loc; ++j)
    {
        for (int k = 0; k < N_ghosts; ++k)
        {
            u_loc[j * Nx_loc + k] = recv_west[k * Ny_loc + j];
            u_loc[j * Nx_loc + Nx_loc - N_ghosts + k] = recv_east[k * Ny_loc + j];
        }
    }

    // Boundary info exchange in y-dir ======================================
    // Send buffer in y-dir
    for (int i = 0; i < Nx_loc; ++i)
    {
        for (int k = 0; k < N_ghosts; ++k)
        {
            send_north[i * N_ghosts + k] = u_loc[(N_ghosts + k) * Nx_loc + i];
            send_south[i * N_ghosts + k] = u_loc[(Ny_loc - 2 * N_ghosts + k) * Nx_loc + i];
        }
    }

    // Exchange Boundary info
    MPI_Sendrecv(send_north, Nx_loc * N_ghosts, MPI_DOUBLE, local_mpi_info->north_rank, 2, recv_south, Nx_loc * N_ghosts, MPI_DOUBLE, local_mpi_info->south_rank, 2, local_mpi_info->comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_south, Nx_loc * N_ghosts, MPI_DOUBLE, local_mpi_info->south_rank, 3, recv_north, Nx_loc * N_ghosts, MPI_DOUBLE, local_mpi_info->north_rank, 3, local_mpi_info->comm, MPI_STATUS_IGNORE);

    // Update local boundary info
    for (int i = 0; i < Nx_loc; ++i)
    {
        for (int k = 0; k < N_ghosts; ++k)
        {
            u_loc[k * Nx_loc + i] = recv_north[i * N_ghosts + k];
            u_loc[(k + Ny_loc - N_ghosts) * Nx_loc + i] = recv_south[i * N_ghosts + k];
        }
    }

    // Deallocation
    delete[] send_west;
    delete[] send_east;
    delete[] send_north;
    delete[] send_south;

    delete[] recv_west;
    delete[] recv_east;
    delete[] recv_north;
    delete[] recv_south;
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
    for (int i = 0; i < Ny; ++i)
    {
        for (int j = 0; j < Nx; ++j)
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

    // Broadcast recvcounts and displs to all processes
    MPI_Bcast(recvcounts, local_mpi_info->world_size, MPI_INT, root, local_mpi_info->comm);
    MPI_Bcast(displs, local_mpi_info->world_size, MPI_INT, root, local_mpi_info->comm);

    for (int j = 0; j < Nx_loc - 2 * N_ghosts; ++j)
    {
        int send_column_idx = (N_ghosts * Ny_loc) + N_ghosts + j * (Ny_loc - 2 * N_ghosts);

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

        MPI_Gatherv(&u_loc[send_column_idx], 1, no_ghost_column_type, u, recvcounts, displs, no_ghost_column_type, root, local_mpi_info->comm);
        MPI_Gatherv(&v_loc[send_column_idx], 1, no_ghost_column_type, v, recvcounts, displs, no_ghost_column_type, root, local_mpi_info->comm);
        MPI_Gatherv(&h_loc[send_column_idx], 1, no_ghost_column_type, h, recvcounts, displs, no_ghost_column_type, root, local_mpi_info->comm);
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
