#include <boost/program_options.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mpi.h>

#include "ShallowWater.h"
#include "cblas.h"

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

void StoreMPIInfo(Local_MPI_Info *local_mpi_info, MPI_Comm comm_cartesian)
{
    const int ndims = 2;
    int periods[ndims] = {1, 1};
    // *local_mpi_info: pointer to the struct Local_MPI_Info
    local_mpi_info->comm = comm_cartesian;
    MPI_Comm_size(comm_cartesian, &(local_mpi_info->world_size));
    MPI_Comm_rank(comm_cartesian, &(local_mpi_info->world_rank));

    // Retrieve virtual topology info
    MPI_Cart_get(comm_cartesian, ndims, local_mpi_info->dims, periods,
                 local_mpi_info->coords);

    // Neighbouring processes
    MPI_Cart_shift(local_mpi_info->comm, 0, 1, &(local_mpi_info->west_rank), &(local_mpi_info->east_rank));
    MPI_Cart_shift(local_mpi_info->comm, 1, 1, &(local_mpi_info->north_rank), &(local_mpi_info->south_rank));
}

void SetInitialConditions(double *u, double *v, double *h, double *h0,
                          int Nx_local, int Ny_local, int Nx, int Ny, int ic,
                          double dx, double dy, Local_MPI_Info *local_mpi_info)
{
    // Starting position coords (globally) of the subdomain (exclusing ghost
    // cells) we have 3 ghost cells each side
    int x_global = local_mpi_info->coords[0] * (Nx_local - 6);
    int y_global = local_mpi_info->coords[1] * (Ny_local - 6);

    for (int i = 0; i < Ny_local; ++i)
    {
        for (int j = 0; j < Nx_local; ++j)
        {
            int i_global = y_global + i - 3; // exclude top 3 ghost cells
            int j_global = x_global + j - 3; // exclude left 3 ghost cells
            // All coded in row-major for now
            u[i * Nx_local + j] = 0.0;
            v[i * Nx_local + j] = 0.0;

            // make sure values not assigned to ghost cells
            if (i_global >= 0 && j_global >= 0 && i_global < Ny && j_global < Nx)
            {
                if (ic == 1)
                {
                    h0[i * Nx_local + j] =
                        10.0 + exp(-(i_global * dx - 50) * (i_global * dx - 50) / 25.0);
                }
                else if (ic == 2)
                {
                    h0[i * Nx_local + j] =
                        10.0 + exp(-(j_global * dy - 50) * (j_global * dy - 50) / 25.0);
                }
                else if (ic == 3)
                {
                    h0[i * Nx_local + j] =
                        10.0 + exp(-((i_global * dx - 50) * (i_global * dx - 50) +
                                     (j_global * dy - 50) * (j_global * dy - 50)) /
                                   25.0);
                }
                else
                {
                    h0[i * Nx_local + j] =
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
    }

    // copy the initial surface height h0 to h as initial conditions
    cblas_dcopy(Nx_local * Ny_local, h0, 1, h, 1);
}

void LocalBCInfoExchange(double *u_local, int Nx_local, int Ny_local, Local_MPI_Info *local_mpi_info)
{
    // ghost cells on each side
    int N_ghost_cells = 3;

    // buffers for sending and receiving data
    // x-dir in cartesian grid
    double *send_west = new double[Ny_local * N_ghost_cells];
    double *recv_west = new double[Ny_local * N_ghost_cells];
    double *send_east = new double[Ny_local * N_ghost_cells];
    double *recv_east = new double[Ny_local * N_ghost_cells];

    // y-dir in cartesian grid
    double *send_north = new double[Nx_local * N_ghost_cells];
    double *recv_north = new double[Nx_local * N_ghost_cells];
    double *send_south = new double[Nx_local * N_ghost_cells];
    double *recv_south = new double[Nx_local * N_ghost_cells];

    // Boundary info exchange in x-dir =====================================
    // send buffer x-dir
    for (int i = 0; i < Ny_local; ++i)
    {
        for (int k = 0; k < N_ghost_cells; ++k)
        {
            send_west[i * N_ghost_cells + k] = u_local[i * (Nx_local + 2 * N_ghost_cells) + N_ghost_cells + k];
            send_east[i * N_ghost_cells + k] = u_local[i * (Nx_local + 2 * N_ghost_cells) + Nx_local - N_ghost_cells + k];
        }
    }

    // Exchange boundary info
    MPI_Sendrecv(send_west, Ny_local * N_ghost_cells, MPI_DOUBLE, local_mpi_info->west_rank, 0, recv_east, Ny_local * N_ghost_cells, MPI_DOUBLE, local_mpi_info->east_rank, 0, local_mpi_info->comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_east, Ny_local * N_ghost_cells, MPI_DOUBLE, local_mpi_info->east_rank, 1, recv_west, Ny_local * N_ghost_cells, MPI_DOUBLE, local_mpi_info->west_rank, 1, local_mpi_info->comm, MPI_STATUS_IGNORE);

    // Update boundary info
    // x-dir
    for (int i = 0; i < Ny_local; ++i)
    {
        for (int k = 0; k < N_ghost_cells; ++k)
        {
            u_local[i * (Nx_local + 2 * N_ghost_cells) + k] = recv_west[i * N_ghost_cells + k];
            u_local[i * (Nx_local + 2 * N_ghost_cells) + Nx_local + N_ghost_cells + k] = recv_east[i * N_ghost_cells + k];
        }
    }

    // Boundary info exchange in y-dir ======================================
    // Send buffer in y-dir
    for (int j = 0; j < Nx_local; ++j)
    {
        for (int k = 0; k < N_ghost_cells; ++k)
        {
            send_north[k * Nx_local + j] = u_local[(k + Ny_local - 2 * N_ghost_cells) * Nx_local + j];
            send_south[k * Nx_local + j] = u_local[(k + N_ghost_cells) * Nx_local + j];
        }
    }

    // Exchange Boundary info
    MPI_Sendrecv(send_north, Nx_local * N_ghost_cells, MPI_DOUBLE, local_mpi_info->north_rank, 2, recv_south, Nx_local * N_ghost_cells, MPI_DOUBLE, local_mpi_info->south_rank, 2, local_mpi_info->comm, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_south, Nx_local * N_ghost_cells, MPI_DOUBLE, local_mpi_info->south_rank, 2, recv_north, Nx_local * N_ghost_cells, MPI_DOUBLE, local_mpi_info->north_rank, 2, local_mpi_info->comm, MPI_STATUS_IGNORE);

    // Update local boundary info
    for (int j = 0; j < Nx_local; ++j)
    {
        for (int k = 0; k < N_ghost_cells; ++k)
        {
            u_local[j + k * Nx_local] = recv_north[j + k * Nx_local];
            u_local[j + (k + Ny_local - N_ghost_cells) * Nx_local] = recv_south[j + k * Nx_local];
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

void SpatialDiscretisation(double *u_local, int Nx_local, int Ny_local, int Nx,
                           int Ny, double dx, double dy, char dir,
                           double *deriv_local, Local_MPI_Info *local_mpi_info)
{

    int N_ghost_cells = 3;

    // Discretisation in x-dir ============================================
    if (dir == 'x')
    {
        double px = 1.0 / dx;

        for (int i = 0; i < Ny_local; ++i)
        {
            for (int j = N_ghost_cells; j < Nx_local + N_ghost_cells; ++j)
            {
                int deriv_idx = i * Nx_local + (j - N_ghost_cells);

                deriv_local[deriv_idx] =
                    px * (-u_local[(i - 3) * (Nx_local + 2 * N_ghost_cells) + j] / 60.0 +
                          3.0 / 20.0 * u_local[(i - 2) * (Nx_local + 2 * N_ghost_cells) + j] -
                          3.0 / 4.0 * u_local[(i - 1) * (Nx_local + 2 * N_ghost_cells) + j] +
                          3.0 / 4.0 * u_local[(i + 1) * (Nx_local + 2 * N_ghost_cells) + j] -
                          3.0 / 20.0 * u_local[(i + 2) * (Nx_local + 2 * N_ghost_cells) + j] +
                          u_local[(i + 3) * (Nx_local + 2 * N_ghost_cells) + j] / 60.0);
            }
        }
    }

    // Discretisation in y-dir ============================================
    else if (dir == 'y')
    {
        double py = 1.0 / dy;

        for (int i = 0; i < Ny_local; ++i)
        {
            for (int j = N_ghost_cells; j < Nx + N_ghost_cells; ++j)
            {
                int deriv_idx = i * Nx_local + (j - N_ghost_cells);
                deriv_local[i * Nx + j] = py * (-u_local[i * (Nx_local + 2 * N_ghost_cells) + j - 3] / 60.0 +
                                                3.0 / 20.0 * u_local[i * (Nx_local + 2 * N_ghost_cells) + (j - 2)] -
                                                3.0 / 4.0 * u_local[i * (Nx_local + 2 * N_ghost_cells) + (j - 1)] +
                                                3.0 / 4.0 * u_local[i * (Nx_local + 2 * N_ghost_cells) + (j + 1)] -
                                                3.0 / 20.0 * u_local[i * (Nx_local + 2 * N_ghost_cells) + (j + 2)] +
                                                u_local[i * (Nx_local + 2 * N_ghost_cells) + (j + 3)] / 60.0);
            }
        }
    }
}

void SpatialDiscretisation_BLAS(double *u, int Nx, int Ny, double dx, double dy,
                                char dir, double *A)
{
    // Banded matrix specs
    const int Kl = 3; // mumber of  subdiags
    const int Ku = 3; // number of superdiags
    const int lda = 1 + Kl + Ku;

    // Discretisation in x-dir ============================================
    if (dir == 'x')
    {
        double px = 1.0 / dx;
        // coefficient of stencil in x-dir
        double coeff_x[7] = {
            -1.0 / 60.0 * px, 3.0 / 20.0 * px, -3.0 / 4.0 * px, 0.0,
            3.0 / 4.0 * px, -3.0 / 20.0 * px, 1.0 / 60.0 * px};

        // Assigning coefficients to the banded matrix
        double *A = new double[lda * Nx];

        for (int i = 0; i < Nx; ++i)
        {
            for (int j = 0; j < lda; ++j)
            {
                A[i * lda + j] = coeff_x[j];
            }
        }
    }

    // Discretisation in y-dir ============================================
    else if (dir == 'y')
    {
        double py = 1.0 / dy;
        // coefficient of stencil in y-dir
        double coeff_y[7] = {
            1.0 / 60.0 * py, -3.0 / 20.0 * py, 3.0 / 4.0 * py, 0.0,
            -3.0 / 4.0 * py, 3.0 / 20.0 * py, -1.0 / 60.0 * py};

        // Assigning coefficients to the banded matrix
        double *A = new double[lda * Ny];

        for (int i = 0; i < Ny; ++i)
        {
            for (int j = 0; j < lda; ++j)
            {
                A[i * lda + j] = coeff_y[j];
            }
        }
    }
}

void Evaluate_fu(double *u_local, double *v_local, double *h_local, int Nx_local, int Ny_local, int Nx, int Ny, double dx, double dy, double *f_local, Local_MPI_Info *local_mpi_info)
{
    double g = 9.81;
    double *deriux_local = new double[Nx_local * Ny_local];
    double *deriuy_local = new double[Nx_local * Ny_local];
    double *derihx_local = new double[Nx_local * Ny_local];

    SpatialDiscretisation(u_local, Nx_local, Ny_local, Nx, Ny, dx, dy, 'x', deriux_local, local_mpi_info);
    SpatialDiscretisation(u_local, Nx_local, Ny_local, Nx, Ny, dx, dy, 'y', deriuy_local, local_mpi_info);
    SpatialDiscretisation(h_local, Nx_local, Ny_local, Nx, Ny, dx, dy, 'x', derihx_local, local_mpi_info);

    for (int i = 0; i < Ny_local; ++i)
    {
        for (int j = 0; j < Nx_local; ++j)
        {
            f_local[i * Nx_local + j] = -u_local[i * Nx_local + j] * deriux_local[i * Nx_local + j] -
                                        v_local[i * Nx_local + j] * deriuy_local[i * Nx_local + j] -
                                        g * derihx_local[i * Nx_local + j];
        }
    }

    delete[] deriux_local;
    delete[] deriuy_local;
    delete[] derihx_local;
}

void Evaluate_fv(double *u_local, double *v_local, double *h_local, int Nx_local, int Ny_local, int Nx, int Ny, double dx, double dy, double *f_local, Local_MPI_Info *local_mpi_info)
{
    double g = 9.81;
    double *derivx_local = new double[Nx_local * Ny_local];
    double *derivy_local = new double[Nx_local * Ny_local];
    double *derihy_local = new double[Nx_local * Ny_local];

    SpatialDiscretisation(v_local, Nx_local, Ny_local, Nx, Ny, dx, dy, 'x', derivx_local, local_mpi_info);
    SpatialDiscretisation(v_local, Nx_local, Ny_local, Nx, Ny, dx, dy, 'y', derivy_local, local_mpi_info);
    SpatialDiscretisation(h_local, Nx_local, Ny_local, Nx, Ny, dx, dy, 'y', derihy_local, local_mpi_info);

    for (int i = 0; i < Ny; ++i)
    {
        for (int j = 0; j < Nx; ++j)
        {
            f_local[i * Nx_local + j] = -u_local[i * Nx_local + j] * derivx_local[i * Nx_local + j] -
                                        v_local[i * Nx_local + j] * derivy_local[i * Nx_local + j] -
                                        g * derihy_local[i * Nx_local + j];
        }
    }

    delete[] derivx_local;
    delete[] derivy_local;
    delete[] derihy_local;
}

void Evaluate_fh(double *u_local, double *v_local, double *h_local, int Nx_local, int Ny_local, int Nx, int Ny, double dx, double dy, double *f_local, Local_MPI_Info *local_mpi_info)
{
    double *derihux_local = new double[Nx_local * Ny_local];
    double *derihvy_local = new double[Nx_local * Ny_local];
    double *hu_local = new double[Nx_local * Ny_local];
    double *hv_local = new double[Nx_local * Ny_local];

    // find hu and hv
    for (int i = 0; i < Ny_local; ++i)
    {
        for (int j = 0; j < Nx_local; ++j)
        {
            hu_local[i * Nx_local + j] = h_local[i * Nx_local + j] * u_local[i * Nx_local + j];
            hv_local[i * Nx_local + j] = h_local[i * Nx_local + j] * v_local[i * Nx_local + j];
        }
    }

    SpatialDiscretisation(hu_local, Nx_local, Ny_local, Nx, Ny, dx, dy, 'x', derihux_local, local_mpi_info);
    SpatialDiscretisation(hv_local, Nx_local, Ny_local, Nx, Ny, dx, dy, 'y', derihvy_local, local_mpi_info);

    for (int i = 0; i < Ny_local; ++i)
    {
        for (int j = 0; j < Nx_local; ++j)
        {
            f_local[i * Nx_local + j] = -derihux_local[i * Nx_local + j] - derihvy_local[i * Nx_local + j];
        }
    }

    delete[] derihux_local;
    delete[] derihvy_local;
    delete[] hu_local;
    delete[] hv_local;
}

// void Evaluate_f_BLAS(double *u, double *v, double *h, int Nx, int Ny, double
// dx, double dy, double *f)
// {
//     double g = 9.81;

//     // Banded matrix specs
//     const int Kl = 3; // mumber of  subdiags
//     const int Ku = 3; // number of superdiags
//     const int lda = 1 + Kl + Ku;

//     // Spatial Discretisations ==========================================
//     double *deriux = new double[Nx * Ny];
//     double *deriuy = new double[Nx * Ny];
//     double *derihx = new double[Nx * Ny];

//     double *derivx = new double[Nx * Ny];
//     double *derivy = new double[Nx * Ny];
//     double *derihy = new double[Nx * Ny];

//     SpatialDiscretisation_BLAS(u, Nx, Ny, dx, dy, 'x', deriux);
//     SpatialDiscretisation_BLAS(u, Nx, Ny, dx, dy, 'y', deriuy);
//     SpatialDiscretisation_BLAS(v, Nx, Ny, dx, dy, 'x', derivx);
//     SpatialDiscretisation_BLAS(v, Nx, Ny, dx, dy, 'y', derivy);
//     SpatialDiscretisation_BLAS(h, Nx, Ny, dx, dy, 'x', derihx);
//     SpatialDiscretisation_BLAS(h, Nx, Ny, dx, dy, 'y', derihy);

//     // Evaluation of fu =================================================
//     for (int j = 0; j < Nx; ++j)
//     {
//     }

//     // Evaluation of fv =================================================

//     // Evaluation of fh =================================================

//     // Deallocations
//     delete[] deriux;
//     delete[] deriuy;
//     delete[] derihx;
//     delete[] derivx;
//     delete[] derivy;
//     delete[] derihy;
// }

void TimeIntegration(double *u_local, double *v_local, double *h_local, int Nx_local, int Ny_local, int Nx, int Ny, double dx, double dy, double dt, double *fu_local, double *fv_local, double *fh_local, Local_MPI_Info *local_mpi_info)
{
    // Solving for u
    double *k1_u_local = new double[Nx_local * Ny_local];
    double *k2_u_local = new double[Nx_local * Ny_local];
    double *k3_u_local = new double[Nx_local * Ny_local];
    double *k4_u_local = new double[Nx_local * Ny_local];

    // Solve for v
    double *k1_v_local = new double[Nx_local * Ny_local];
    double *k2_v_local = new double[Nx_local * Ny_local];
    double *k3_v_local = new double[Nx_local * Ny_local];
    double *k4_v_local = new double[Nx_local * Ny_local];

    // Solve for h
    double *k1_h_local = new double[Nx_local * Ny_local];
    double *k2_h_local = new double[Nx_local * Ny_local];
    double *k3_h_local = new double[Nx_local * Ny_local];
    double *k4_h_local = new double[Nx_local * Ny_local];

    double *tu_local = new double[Nx_local * Ny_local]; // temp vector t = u
    double *tv_local = new double[Nx_local * Ny_local]; // temp vector t = v
    double *th_local = new double[Nx_local * Ny_local]; // temp vector t = h

    // Calculating k1 = f(yn) ===================================
    cblas_dcopy(Nx_local * Ny_local, u_local, 1, tu_local, 1);
    cblas_dcopy(Nx_local * Ny_local, v_local, 1, tv_local, 1);
    cblas_dcopy(Nx_local * Ny_local, h_local, 1, th_local, 1);

    // Exchange boundary info between each process
    LocalBCInfoExchange(u_local, Nx_local, Ny_local, local_mpi_info);
    LocalBCInfoExchange(v_local, Nx_local, Ny_local, local_mpi_info);
    LocalBCInfoExchange(h_local, Nx_local, Ny_local, local_mpi_info);

    Evaluate_fu(u_local, v_local, h_local, Nx_local, Ny_local, Nx, Ny, dx, dy, fu_local, local_mpi_info);
    Evaluate_fv(u_local, v_local, h_local, Nx_local, Ny_local, Nx, Ny, dx, dy, fv_local, local_mpi_info);
    Evaluate_fh(u_local, v_local, h_local, Nx_local, Ny_local, Nx, Ny, dx, dy, fh_local, local_mpi_info);

    // Evaluate_fu_BLAS(u, v, h, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(u, v, h, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(u, v, h, Nx, Ny, dx, dy, fh);

    cblas_dcopy(Nx_local * Ny_local, fu_local, 1, k1_u_local, 1);
    cblas_dcopy(Nx_local * Ny_local, fv_local, 1, k1_v_local, 1);
    cblas_dcopy(Nx_local * Ny_local, fh_local, 1, k1_h_local, 1);

    // Calculating k2 = f(yn + dt*k1/2) ==========================
    // reset temp values
    cblas_dcopy(Nx_local * Ny_local, u_local, 1, tu_local, 1); // reset tu to u
    cblas_dcopy(Nx_local * Ny_local, v_local, 1, tv_local, 1);
    cblas_dcopy(Nx_local * Ny_local, h_local, 1, th_local, 1);

    // update un to un+dt*k1/2 to evaluate f for k2
    cblas_daxpy(Nx_local * Ny_local, dt / 2.0, k1_u_local, 1, tu_local, 1);
    cblas_daxpy(Nx_local * Ny_local, dt / 2.0, k1_v_local, 1, tv_local, 1);
    cblas_daxpy(Nx_local * Ny_local, dt / 2.0, k1_h_local, 1, th_local, 1);

    // Exchange boundary info between each process
    LocalBCInfoExchange(tu_local, Nx_local, Ny_local, local_mpi_info);
    LocalBCInfoExchange(tv_local, Nx_local, Ny_local, local_mpi_info);
    LocalBCInfoExchange(th_local, Nx_local, Ny_local, local_mpi_info);

    // Evaluate new f
    Evaluate_fu(tu_local, tv_local, th_local, Nx_local, Ny_local, Nx, Ny, dx, dy, fu_local, local_mpi_info);
    Evaluate_fv(tu_local, tv_local, th_local, Nx_local, Ny_local, Nx, Ny, dx, dy, fv_local, local_mpi_info);
    Evaluate_fh(tu_local, tv_local, th_local, Nx_local, Ny_local, Nx, Ny, dx, dy, fh_local, local_mpi_info);

    // Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

    cblas_dcopy(Nx_local * Ny_local, fu_local, 1, k2_u_local, 1);
    cblas_dcopy(Nx_local * Ny_local, fv_local, 1, k2_v_local, 1);
    cblas_dcopy(Nx_local * Ny_local, fh_local, 1, k2_h_local, 1);

    // Calculating k3 = f(yn+dt*k2/2) =============================
    // reset temp values
    cblas_dcopy(Nx_local * Ny_local, u_local, 1, tu_local, 1); // reset tu to u
    cblas_dcopy(Nx_local * Ny_local, v_local, 1, tv_local, 1);
    cblas_dcopy(Nx_local * Ny_local, h_local, 1, th_local, 1);

    // update un to un+dt*k2/2 to evaluate f for k3
    cblas_daxpy(Nx_local * Ny_local, dt / 2.0, k2_u_local, 1, tu_local, 1);
    cblas_daxpy(Nx_local * Ny_local, dt / 2.0, k2_v_local, 1, tv_local, 1);
    cblas_daxpy(Nx_local * Ny_local, dt / 2.0, k2_h_local, 1, th_local, 1);

    // Exchange boundary info between each process
    LocalBCInfoExchange(tu_local, Nx_local, Ny_local, local_mpi_info);
    LocalBCInfoExchange(tv_local, Nx_local, Ny_local, local_mpi_info);
    LocalBCInfoExchange(th_local, Nx_local, Ny_local, local_mpi_info);

    Evaluate_fu(tu_local, tv_local, th_local, Nx_local, Ny_local, Nx, Ny, dx, dy, fu_local, local_mpi_info);
    Evaluate_fv(tu_local, tv_local, th_local, Nx_local, Ny_local, Nx, Ny, dx, dy, fv_local, local_mpi_info);
    Evaluate_fh(tu_local, tv_local, th_local, Nx_local, Ny_local, Nx, Ny, dx, dy, fh_local, local_mpi_info);

    // Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

    cblas_dcopy(Nx_local * Ny_local, fu_local, 1, k3_u_local, 1);
    cblas_dcopy(Nx_local * Ny_local, fv_local, 1, k3_v_local, 1);
    cblas_dcopy(Nx_local * Ny_local, fh_local, 1, k3_h_local, 1);

    // k4 = f(yn+dt*k3) ===========================================
    // reset temp values
    cblas_dcopy(Nx_local * Ny_local, u_local, 1, tu_local, 1); // reset tu to u
    cblas_dcopy(Nx_local * Ny_local, v_local, 1, tv_local, 1);
    cblas_dcopy(Nx_local * Ny_local, h_local, 1, th_local, 1);

    // update un to un+dt*k2/2 to evaluate f for k3
    cblas_daxpy(Nx_local * Ny_local, dt, k3_u_local, 1, tu_local, 1);
    cblas_daxpy(Nx_local * Ny_local, dt, k3_v_local, 1, tv_local, 1);
    cblas_daxpy(Nx_local * Ny_local, dt, k3_h_local, 1, th_local, 1);

    // Exchange boundary info between each process
    LocalBCInfoExchange(tu_local, Nx_local, Ny_local, local_mpi_info);
    LocalBCInfoExchange(tv_local, Nx_local, Ny_local, local_mpi_info);
    LocalBCInfoExchange(th_local, Nx_local, Ny_local, local_mpi_info);

    Evaluate_fu(tu_local, tv_local, th_local, Nx_local, Ny_local, Nx, Ny, dx, dy, fu_local, local_mpi_info);
    Evaluate_fv(tu_local, tv_local, th_local, Nx_local, Ny_local, Nx, Ny, dx, dy, fv_local, local_mpi_info);
    Evaluate_fh(tu_local, tv_local, th_local, Nx_local, Ny_local, Nx, Ny, dx, dy, fh_local, local_mpi_info);

    // Evaluate_fu_BLAS(tu, tv, th, Nx, Ny, dx, dy, fu);
    // Evaluate_fv_BLAS(tu, tv, th, Nx, Ny, dx, dy, fv);
    // Evaluate_fh_BLAS(tu, tv, th, Nx, Ny, dx, dy, fh);

    cblas_dcopy(Nx_local * Ny_local, fu_local, 1, k4_u_local, 1);
    cblas_dcopy(Nx_local * Ny_local, fv_local, 1, k4_v_local, 1);
    cblas_dcopy(Nx_local * Ny_local, fh_local, 1, k4_h_local, 1);

    // yn+1 = yn + 1/6*(k1+2*k2+2*k3+k4)*dt
    // Update solution
    for (int i = 0; i < Ny_local; ++i)
    {
        for (int j = 0; j < Nx_local; ++j)
        {
            u_local[i * Nx + j] += dt / 6.0 *
                                   (k1_u_local[i * Nx + j] + 2.0 * k2_u_local[i * Nx + j] +
                                    2.0 * k3_u_local[i * Nx + j] + k4_u_local[i * Nx + j]);
            v_local[i * Nx + j] += dt / 6.0 *
                                   (k1_v_local[i * Nx + j] + 2.0 * k2_v_local[i * Nx + j] +
                                    2.0 * k3_v_local[i * Nx + j] + k4_v_local[i * Nx + j]);
            h_local[i * Nx + j] += dt / 6.0 *
                                   (k1_h_local[i * Nx + j] + 2.0 * k2_h_local[i * Nx + j] +
                                    2.0 * k3_h_local[i * Nx + j] + k4_h_local[i * Nx + j]);
        }
    }

    // deallocate memory
    delete[] k1_u_local;
    delete[] k2_u_local;
    delete[] k3_u_local;
    delete[] k4_u_local;

    delete[] k1_v_local;
    delete[] k2_v_local;
    delete[] k3_v_local;
    delete[] k4_v_local;

    delete[] k1_h_local;
    delete[] k2_h_local;
    delete[] k3_h_local;
    delete[] k4_h_local;
}

int main(int argc, char *argv[])
{
    cout << "Goodbye World" << endl;

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
        cout << options << endl;
    }

    // Assign parameters
    const double dt = vm["dt"].as<double>();
    const double T = vm["T"].as<double>();
    const int Nx = vm["Nx"].as<int>();
    const int Ny = vm["Ny"].as<int>();
    const int ic = vm["ic"].as<int>();

    // // debug output
    // cout << dt << endl;
    // cout << T << endl;
    // cout << Nx << endl;
    // cout << Ny << endl;
    // cout << ic << endl;

    // calculating dx and dy
    const double dx = 1.0;
    const double dy = 1.0;

    // MPI ==================================================
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

    // Cartesian Topology
    const int ndims = 2;
    int dims[ndims] = {0, 0};
    int periods[ndims] = {1, 1};
    MPI_Comm comm_cartesian;
    MPI_Dims_create(world_size, ndims, dims); // automatic division in grid
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, 1, &comm_cartesian);

    // Store MPI Information
    Local_MPI_Info local_mpi_info;
    StoreMPIInfo(&local_mpi_info, comm_cartesian);

    // // Current rank and coordinates
    // int cart_rank;
    // int coords[ndims];
    // MPI_Comm_rank(comm_cartesian, &cart_rank);
    // MPI_Cart_coords(comm_cartesian, cart_rank, ndims, coords);
    // printf("Rank %d/%d: Cartesian coordinates: (%d, %d), Grid dimensions: %d x
    // %d\n", world_rank, world_size, coords[0], coords[1], dims[0], dims[1]);

    // // Determine the neighboring process
    // int left, right, top, bottom;
    // MPI_Cart_shift(comm_cartesian, 0, 1, &left, &right);
    // MPI_Cart_shift(comm_cartesian, 1, 1, &top, &bottom);
    // printf("Rank %d: left = %d, right = %d, top = %d, bottom = %d\n",
    //        world_rank, left, right, top, bottom);

    // Subdomain ===============================================
    int Nx_local = Nx / dims[0] + 6; // 3 ghost cells each side
    int Ny_local = Ny / dims[1] + 6; // 3 ghost cells each side

    // Allocate solution memories (allocated with ghost cells)
    double *u_local = new double[Nx_local * Ny_local];
    double *v_local = new double[Nx_local * Ny_local];
    double *h_local = new double[Nx_local * Ny_local];
    double *h0_local = new double[Nx_local * Ny_local];

    // =========================================================
    // test for SetInitialConditions
    SetInitialConditions(u_local, v_local, h_local, h0_local, Nx_local,
                         Ny_local, Nx, Ny, ic, dx, dy, &local_mpi_info);

    // // debug output
    // cout << "====== h ======" << endl;
    // printMatrix(Nx,Ny,h);

    // ======================================================
    // test for evaluating f
    double *fu_local = new double[Nx_local * Ny_local];
    double *fv_local = new double[Nx_local * Ny_local];
    double *fh_local = new double[Nx_local * Ny_local];
    // // Evaluate_fu_BLAS(u, v, h, Nx, Ny, dx, dy, fu);
    // // Evaluate_fv_BLAS(u, v, h, Nx, Ny, dx, dy, fv);
    // // Evaluate_fh_BLAS(u, v, h, Nx, Ny, dx, dy, fh);

    // ======================================================
    // 4th order RK Time Integrations

    // Time advancement
    double time = 0.0; // start time
    while (time <= T)
    {
        TimeIntegration(u_local, v_local, h_local, Nx_local, Ny_local, Nx, Ny, dx, dy, dt, fu_local, fv_local, fh_local, &local_mpi_info);
        time += dt;
    }

    // // for (int i = 0; i < 10; ++i)
    // // {
    // //     TimeIntegration(u, v, h, Nx, Ny, dx, dy, dt, T, fu, fv, fh);
    // // }

    // // TimeIntegration(u, v, h, Nx, Ny, dx, dy, dt, fu, fv, fh);
    // // TimeIntegration(u, v, h, Nx, Ny, dx, dy, dt, fu, fv, fh);
    // // TimeIntegration(u, v, h, Nx, Ny, dx, dy, dt, fu, fv, fh);
    // // TimeIntegration(u, v, h, Nx, Ny, dx, dy, dt, fu, fv, fh);
    // // TimeIntegration(u, v, h, Nx, Ny, dx, dy, dt, fu, fv, fh);
    // // TimeIntegration(u, v, h, Nx, Ny, dx, dy, dt, fu, fv, fh);

    // // ======================================================
    // // Write to file
    // // Write initial condition
    // ofstream vOut("output.txt", ios::out | ios ::trunc);
    // vOut.precision(5);
    // for (int j = 0; j < Nx; ++j)
    // {
    //     for (int i = 0; i < Ny; ++i)
    //     {
    //         vOut << setw(15) << i * dx << setw(15) << j * dy << setw(15) << u[i
    //         * Nx + j] << setw(15) << v[i * Nx + j] << setw(15) << h[i * Nx + j]
    //         << endl;
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