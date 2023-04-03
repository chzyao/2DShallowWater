#include "ShallowWater.h"
#include "Comm.h"
#include <boost/program_options.hpp>
#include <cblas.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <omp.h>

using namespace std;
namespace po = boost::program_options;

ShallowWater::ShallowWater() 
{
}

void ShallowWater::SetParameters(int argc, char *argv[])
{
    // Read parameters from command line =========================
    po::options_description options("Available Options.");
    options.add_options()("help", "Display help message")(
        "dt", po::value<double>(&m_dt)->default_value(DEFAULT_DT),
        "Time-step to use")("T",
                            po::value<double>(&m_T)->default_value(DEFAULT_T),
                            "Total integration time")(
        "Nx", po::value<int>(&m_Nx)->default_value(DEFAULT_NX),
        "Number of grid points in x")(
        "Ny", po::value<int>(&m_Ny)->default_value(DEFAULT_NY),
        "Number of grid points in y")(
        "ic", po::value<int>(&m_ic)->default_value(DEFAULT_IC),
        "Index of the initial condition to use (1-4)")(
        "method", po::value<char>(&m_method)->default_value(DEFAULT_METHOD),
        "Evaluation of f Method ('l': Loop, 'b': BLAS)");

    po::variables_map vm;

    try
    {
        po::store(po::parse_command_line(argc, argv, options), vm);

        // Display help message
        if (vm.count("help"))
        {
            cout << options << endl;
            exit(EXIT_SUCCESS);
        }

        po::notify(vm);
    }

    catch (const po::error &e)
    {
        cerr << "Error: " << e.what() << endl;
        cerr << options << endl;
        exit(EXIT_FAILURE);
    }

    // Mesh Sizes
    m_dx = 1.0;
    m_dy = 1.0;
}

void ShallowWater::SetInitialConditions(Comm::MPI_Info *mpi_info)
{
    // Generate Initial conditions in root rank
    if (mpi_info->m_rank == 0)
    {

#pragma omp parallel for collapse(2)
        for (int i = 0; i < m_Nx; ++i)
        {
            for (int j = 0; j < m_Ny; ++j)
            {
                // All coded in row-major for now
                m_u[i * m_Ny + j] = 0.0;
                m_v[i * m_Ny + j] = 0.0;
                if (m_ic == 1)
                {
                    m_h0[i * m_Ny + j] =
                        10.0 + exp(-(i * m_dx - 50) * (i * m_dx - 50) / 25.0);
                }
                else if (m_ic == 2)
                {
                    m_h0[i * m_Ny + j] =
                        10.0 + exp(-(j * m_dy - 50) * (j * m_dy - 50) / 25.0);
                }
                else if (m_ic == 3)
                {
                    m_h0[i * m_Ny + j] = 10.0 + exp(-((i * m_dx - 50) * (i * 
                                                       m_dx - 50) +
                                                      (j * m_dy - 50) * (j * m_dy - 50)) /
                                                    25.0);
                }
                else
                {
                    m_h0[i * m_Ny + j] = 10.0 +
                                         exp(-((i * m_dx - 25) * (i * m_dx - 25) +
                                               (j * m_dy - 25) * (j * m_dy - 25)) /
                                             25.0) +
                                         exp(-((i * m_dx - 75) * (i * m_dx - 75) +
                                               (j * m_dy - 75) * (j * m_dy - 75)) /
                                             25.0);
                }
            }
        }

        // copy the initial surface height h0 to h as initial conditions
        cblas_dcopy(m_Nx * m_Ny, m_h0, 1, m_h, 1);
    }

    // Scatter to each rank
    MPI_Scatterv(m_u, mpi_info->sendcounts, mpi_info->displs, MPI_DOUBLE, m_u_loc,
                 m_Nx * m_Ny_loc, MPI_DOUBLE, 0, mpi_info->m_comm);
    MPI_Scatterv(m_v, mpi_info->sendcounts, mpi_info->displs, MPI_DOUBLE, m_v_loc,
                 m_Nx * m_Ny_loc, MPI_DOUBLE, 0, mpi_info->m_comm);
    MPI_Scatterv(m_h, mpi_info->sendcounts, mpi_info->displs, MPI_DOUBLE, m_h_loc,
                 m_Nx * m_Ny_loc, MPI_DOUBLE, 0, mpi_info->m_comm);
}

void ShallowWater::SpatialDiscretisation(double *u, double *u_loc, char dir,
                                         double *deriv, double *deriv_loc,
                                         Comm::MPI_Info *mpi_info)
{
    // Gather results of u
    MPI_Gatherv(u_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, u, mpi_info->recvcounts,
                mpi_info->displs, MPI_DOUBLE, 0, mpi_info->m_comm);

    if (mpi_info->m_rank == 0)
    {
        // LOOP Based Approach ================================
        if (m_method == 'l')
        {
            // Discretisation in x-dir ============================================
            if (dir == 'x')
            {
                double px = 1.0 / m_dx;

                double coeff_x[7] = {
                    -1.0 / 60.0 * px, 3.0 / 20.0 * px, -3.0 / 4.0 * px, 0.0,
                    3.0 / 4.0 * px, -3.0 / 20.0 * px, 1.0 / 60.0 * px};

#pragma omp parallel for collapse(2)
                for (int i = 0; i < m_Nx; ++i)
                {
                    for (int j = 0; j < m_Ny; ++j)
                    {
                        int i_minus_1 = i - 1;
                        int i_minus_2 = i - 2;
                        int i_minus_3 = i - 3;
                        int i_plus_1 = i + 1;
                        int i_plus_2 = i + 2;
                        int i_plus_3 = i + 3;

                        if (i_minus_1 < 0)
                            i_minus_1 += m_Nx;
                        if (i_minus_2 < 0)
                            i_minus_2 += m_Nx;
                        if (i_minus_3 < 0)
                            i_minus_3 += m_Nx;
                        if (i_plus_1 >= m_Nx)
                            i_plus_1 -= m_Nx;
                        if (i_plus_2 >= m_Nx)
                            i_plus_2 -= m_Nx;
                        if (i_plus_3 >= m_Nx)
                            i_plus_3 -= m_Nx;

                        deriv[i * m_Ny + j] = (coeff_x[0] * u[i_minus_3 * m_Ny + j] +
                                               coeff_x[1] * u[i_minus_2 * m_Ny + j] +
                                               coeff_x[2] * u[i_minus_1 * m_Ny + j] +
                                               coeff_x[4] * u[i_plus_1 * m_Ny + j] +
                                               coeff_x[5] * u[i_plus_2 * m_Ny + j] +
                                               coeff_x[6] * u[i_plus_3 * m_Ny + j]);
                    }
                }
            }

            // Discretisation in y-dir ============================================
            else if (dir == 'y')
            {
                double py = 1.0 / m_dy;

                double coeff_y[7] = {
                    -1.0 / 60.0 * py, 3.0 / 20.0 * py, -3.0 / 4.0 * py, 0.0,
                    3.0 / 4.0 * py, -3.0 / 20.0 * py, 1.0 / 60.0 * py};

#pragma omp parallel for collapse(2)
                for (int i = 0; i < m_Nx; ++i)
                {

                    for (int j = 0; j < m_Ny; ++j)
                    {
                        int j_minus_1 = j - 1;
                        int j_minus_2 = j - 2;
                        int j_minus_3 = j - 3;
                        int j_plus_1 = j + 1;
                        int j_plus_2 = j + 2;
                        int j_plus_3 = j + 3;

                        if (j_minus_1 < 0)
                            j_minus_1 += m_Ny;
                        if (j_minus_2 < 0)
                            j_minus_2 += m_Ny;
                        if (j_minus_3 < 0)
                            j_minus_3 += m_Ny;
                        if (j_plus_1 >= m_Ny)
                            j_plus_1 -= m_Ny;
                        if (j_plus_2 >= m_Ny)
                            j_plus_2 -= m_Ny;
                        if (j_plus_3 >= m_Ny)
                            j_plus_3 -= m_Ny;

                        deriv[i * m_Ny + j] = (coeff_y[0] * u[i * m_Ny + j_minus_3] +
                                               coeff_y[1] * u[i * m_Ny + j_minus_2] +
                                               coeff_y[2] * u[i * m_Ny + j_minus_1] +
                                               coeff_y[4] * u[i * m_Ny + j_plus_1] +
                                               coeff_y[5] * u[i * m_Ny + j_plus_2] +
                                               coeff_y[6] * u[i * m_Ny + j_plus_3]);
                    }
                }
            }
        }

        // BLAS based Approach ================================================
        else if (m_method == 'b')
        {
            const int ku = 6;            // superdiags
            const int kl = 0;            // subdiags
            const int lda = 1 + ku + kl; // leading dimensions
            double *A;                   // Banded Matrix to be filled

            // Note that although it is possible to implement this method with 3 upper
            // and lower diagonals, it could be miles easier to enforce the periodic
            // boundary conditions using only 6 superdiagonals. Storage-wise, not much
            // of a huge difference as it is only 6*lda slots larger

            // Discretisation in x-dir =======================
            if (dir == 'x')
            {
                A = new double[lda * (m_Nx + 6)];
                double px = 1.0 / m_dx;
                // coefficient of stencil in x-dir
                double coeff_x[7] = {
                    1.0 / 60.0 * px, -3.0 / 20.0 * px, 3.0 / 4.0 * px, 0.0,
                    -3.0 / 4.0 * px, 3.0 / 20.0 * px, -1.0 / 60.0 * px};

                for (int i = 0; i < m_Nx + 6; ++i)
                {
                    A[i * lda] = coeff_x[0];     // original upper diag 1
                    A[i * lda + 1] = coeff_x[1]; // original upper diag 2
                    A[i * lda + 2] = coeff_x[2]; // original upper diag 3
                    A[i * lda + 3] = coeff_x[3]; // original diag
                    A[i * lda + 4] = coeff_x[4]; // original lower diag 1
                    A[i * lda + 5] = coeff_x[5]; // original lower diag 2
                    A[i * lda + 6] = coeff_x[6]; // original lower diag 3
                }

#pragma omp parallel
                {
                    // Temp vector to store the column element of u
                    double *u_col = new double[m_Nx + 6];
                    // Temp vector to store the column element of deriv
                    double *deriv_col = new double[m_Nx + 6];

#pragma omp for
                    // BLAS dgbmv and for loop to find deriv
                    for (int j = 0; j < m_Ny; ++j)
                    {
                        for (int i = 0; i < m_Nx; ++i)
                        {
                            u_col[i + 3] = u[i * m_Ny + j];
                        }
                        // Handling periodic BC
                        u_col[0] = u_col[m_Nx];
                        u_col[1] = u_col[m_Nx + 1];
                        u_col[2] = u_col[m_Nx + 2];
                        u_col[m_Nx + 3] = u_col[3];
                        u_col[m_Nx + 4] = u_col[4];
                        u_col[m_Nx + 5] = u_col[5];

                        cblas_dgbmv(CblasColMajor, CblasNoTrans, m_Nx + 6, m_Nx + 6, kl, ku,
                                    1.0, A, lda, u_col, 1, 0.0, deriv_col, 1);

                        for (int i = 0; i < m_Nx; ++i)
                        {
                            deriv[i * m_Ny + j] = deriv_col[i];
                        }
                    }

                    // Deallocations
                    delete[] u_col;
                    delete[] deriv_col;
                }
            }

            // Discretisation in y-dir ============================================
            else if (dir == 'y')
            {
                double py = 1.0 / m_dy;
                A = new double[lda * (m_Ny + 6)];

                // coefficient of stencil in y-dir
                double coeff_y[7] = {
                    -1.0 / 60.0 * py, 3.0 / 20.0 * py, -3.0 / 4.0 * py, 0.0,
                    3.0 / 4.0 * py, -3.0 / 20.0 * py, 1.0 / 60.0 * py};

                for (int i = 0; i < m_Ny + 6; ++i)
                {
                    A[i * lda] = coeff_y[0];     // original upper diag 1
                    A[i * lda + 1] = coeff_y[1]; // original upper diag 2
                    A[i * lda + 2] = coeff_y[2]; // original upper diag 3
                    A[i * lda + 3] = coeff_y[3]; // original diag
                    A[i * lda + 4] = coeff_y[4]; // original lower diag 1
                    A[i * lda + 5] = coeff_y[5]; // original lower diag 2
                    A[i * lda + 6] = coeff_y[6]; // original lower diag 3
                }
#pragma omp parallel
                {
                    // Temp vector to store the column element of u
                    double *u_col = new double[m_Ny + 6];
                    // Temp vector to store the column element of deriv
                    double *deriv_col = new double[m_Ny + 6];

#pragma omp for
                    // BLAS dgbmv and for loop to find deriv
                    for (int i = 0; i < m_Nx; ++i)
                    {
                        for (int j = 0; j < m_Ny; ++j)
                        {
                            u_col[j + 3] = u[i * m_Ny + j];
                        }
                        // Handling periodic BC
                        u_col[0] = u_col[m_Ny];
                        u_col[1] = u_col[m_Ny + 1];
                        u_col[2] = u_col[m_Ny + 2];
                        u_col[m_Ny + 3] = u_col[3];
                        u_col[m_Ny + 4] = u_col[4];
                        u_col[m_Ny + 5] = u_col[5];

                        cblas_dgbmv(CblasColMajor, CblasNoTrans, m_Ny + 6, m_Ny + 6, kl, ku,
                                    1.0, A, lda, u_col, 1, 0.0, deriv_col, 1);

                        for (int j = 0; j < m_Ny; ++j)
                        {
                            deriv[i * m_Ny + j] = deriv_col[j];
                        }
                    }

                    // Deallocations
                    delete[] u_col;
                    delete[] deriv_col;
                }
            }

            delete[] A;
        }
    }

    // Scatter to each process
    MPI_Scatterv(deriv, mpi_info->sendcounts, mpi_info->displs, MPI_DOUBLE,
                 deriv_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, 0, mpi_info->m_comm);
}

void ShallowWater::Evaluate_f(double *u, double *v, double *h, double *u_loc,
                              double *v_loc, double *h_loc, double *fu_loc,
                              double *fv_loc, double *fh_loc,
                              Comm::MPI_Info *mpi_info)
{
    double g = 9.81;

    double *deriux = nullptr;
    double *deriuy = nullptr;

    double *derivx = nullptr;
    double *derivy = nullptr;

    double *derihx = nullptr;
    double *derihy = nullptr;

    // Derivatives of u
    if (mpi_info->m_rank == 0)
    {
        deriux = new double[m_Nx * m_Ny];
        deriuy = new double[m_Nx * m_Ny];

        derivx = new double[m_Nx * m_Ny];
        derivy = new double[m_Nx * m_Ny];

        derihx = new double[m_Nx * m_Ny];
        derihy = new double[m_Nx * m_Ny];
    }

    // Derivatives of u
    double *deriux_loc = new double[m_Nx * m_Ny_loc];
    double *deriuy_loc = new double[m_Nx * m_Ny_loc];

    // Derivatives of v
    double *derivx_loc = new double[m_Nx * m_Ny_loc];
    double *derivy_loc = new double[m_Nx * m_Ny_loc];

    // Derivatives of h

    double *derihx_loc = new double[m_Nx * m_Ny_loc];
    double *derihy_loc = new double[m_Nx * m_Ny_loc];

    SpatialDiscretisation(u, u_loc, 'x', deriux, deriux_loc, mpi_info);
    SpatialDiscretisation(u, u_loc, 'y', deriuy, deriuy_loc, mpi_info);

    SpatialDiscretisation(v, v_loc, 'x', derivx, derivx_loc, mpi_info);
    SpatialDiscretisation(v, v_loc, 'y', derivy, derivy_loc, mpi_info);

    SpatialDiscretisation(h, h_loc, 'x', derihx, derihx_loc, mpi_info);
    SpatialDiscretisation(h, h_loc, 'y', derihy, derihy_loc, mpi_info);

    // RHS terms are evaluated locally in each rank
    for (int i = 0; i < m_Nx; ++i)
    {
        for (int j = 0; j < m_Ny_loc; ++j)
        {
            fu_loc[i * m_Ny_loc + j] =
                -u_loc[i * m_Ny_loc + j] * deriux_loc[i * m_Ny_loc + j] -
                v_loc[i * m_Ny_loc + j] * deriuy_loc[i * m_Ny_loc + j] -
                g * derihx_loc[i * m_Ny_loc + j];

            fv_loc[i * m_Ny_loc + j] =
                -u_loc[i * m_Ny_loc + j] * derivx_loc[i * m_Ny_loc + j] -
                v_loc[i * m_Ny_loc + j] * derivy_loc[i * m_Ny_loc + j] -
                g * derihy_loc[i * m_Ny_loc + j];

            fh_loc[i * m_Ny_loc + j] =
                -h_loc[i * m_Ny_loc + j] * deriux_loc[i * m_Ny_loc + j] -
                u_loc[i * m_Ny_loc + j] * derihx_loc[i * m_Ny_loc + j] -
                h_loc[i * m_Ny_loc + j] * derivy_loc[i * m_Ny_loc + j] -
                v_loc[i * m_Ny_loc + j] * derihy_loc[i * m_Ny_loc + j];
        }
    }

    if (mpi_info->m_rank == 0)
    {
        delete[] deriux;
        delete[] deriuy;

        delete[] derivx;
        delete[] derivy;

        delete[] derihx;
        delete[] derihy;
    }

    delete[] deriux_loc;
    delete[] deriuy_loc;

    delete[] derivx_loc;
    delete[] derivy_loc;

    delete[] derihx_loc;
    delete[] derihy_loc;
}

void ShallowWater::TimeIntegration(double *u_loc, double *v_loc, double *h_loc,
                                   double *fu_loc, double *fv_loc,
                                   double *fh_loc, Comm::MPI_Info *mpi_info)
{
    // Solving for u
    double *k1_u = new double[m_Nx * m_Ny_loc];
    double *k2_u = new double[m_Nx * m_Ny_loc];
    double *k3_u = new double[m_Nx * m_Ny_loc];
    double *k4_u = new double[m_Nx * m_Ny_loc];

    // Solve for v
    double *k1_v = new double[m_Nx * m_Ny_loc];
    double *k2_v = new double[m_Nx * m_Ny_loc];
    double *k3_v = new double[m_Nx * m_Ny_loc];
    double *k4_v = new double[m_Nx * m_Ny_loc];

    // Solve for h
    double *k1_h = new double[m_Nx * m_Ny_loc];
    double *k2_h = new double[m_Nx * m_Ny_loc];
    double *k3_h = new double[m_Nx * m_Ny_loc];
    double *k4_h = new double[m_Nx * m_Ny_loc];

    double *tu_loc = new double[m_Nx * m_Ny_loc]; // temp vector t = u
    double *tv_loc = new double[m_Nx * m_Ny_loc]; // temp vector t = v
    double *th_loc = new double[m_Nx * m_Ny_loc]; // temp vector t = h

    double *tu = nullptr;
    double *tv = nullptr;
    double *th = nullptr;

    if (mpi_info->m_rank == 0)
    {
        tu = new double[m_Nx * m_Ny];
        tv = new double[m_Nx * m_Ny];
        th = new double[m_Nx * m_Ny];
    }

    // Calculating k1 = f(yn) ===================================
    cblas_dcopy(m_Nx * m_Ny_loc, u_loc, 1, tu_loc, 1);
    cblas_dcopy(m_Nx * m_Ny_loc, v_loc, 1, tv_loc, 1);
    cblas_dcopy(m_Nx * m_Ny_loc, h_loc, 1, th_loc, 1);

    MPI_Gatherv(tu_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, tu, mpi_info->recvcounts,
                mpi_info->displs, MPI_DOUBLE, 0, mpi_info->m_comm);
    MPI_Gatherv(tv_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, tv, mpi_info->recvcounts,
                mpi_info->displs, MPI_DOUBLE, 0, mpi_info->m_comm);
    MPI_Gatherv(th_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, th, mpi_info->recvcounts,
                mpi_info->displs, MPI_DOUBLE, 0, mpi_info->m_comm);

    Evaluate_f(tu, tv, th, tu_loc, tv_loc, th_loc, fu_loc, fv_loc, fh_loc,
               mpi_info);

    cblas_dcopy(m_Nx * m_Ny_loc, fu_loc, 1, k1_u, 1);
    cblas_dcopy(m_Nx * m_Ny_loc, fv_loc, 1, k1_v, 1);
    cblas_dcopy(m_Nx * m_Ny_loc, fh_loc, 1, k1_h, 1);

    // Calculating k2 = f(yn + dt*k1/2) ==========================
    // reset temp values
    cblas_dcopy(m_Nx * m_Ny_loc, m_u_loc, 1, tu_loc, 1); // reset tu to u
    cblas_dcopy(m_Nx * m_Ny_loc, m_v_loc, 1, tv_loc, 1);
    cblas_dcopy(m_Nx * m_Ny_loc, m_h_loc, 1, th_loc, 1);

    // update un to un+dt*k1/2 to evaluate f for k2
    cblas_daxpy(m_Nx * m_Ny_loc, m_dt / 2.0, k1_u, 1, tu_loc, 1);
    cblas_daxpy(m_Nx * m_Ny_loc, m_dt / 2.0, k1_v, 1, tv_loc, 1);
    cblas_daxpy(m_Nx * m_Ny_loc, m_dt / 2.0, k1_h, 1, th_loc, 1);

    MPI_Gatherv(tu_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, tu, mpi_info->recvcounts,
                mpi_info->displs, MPI_DOUBLE, 0, mpi_info->m_comm);
    MPI_Gatherv(tv_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, tv, mpi_info->recvcounts,
                mpi_info->displs, MPI_DOUBLE, 0, mpi_info->m_comm);
    MPI_Gatherv(th_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, th, mpi_info->recvcounts,
                mpi_info->displs, MPI_DOUBLE, 0, mpi_info->m_comm);

    // Evaluate new f
    Evaluate_f(tu, tv, th, tu_loc, tv_loc, th_loc, fu_loc, fv_loc, fh_loc,
               mpi_info);

    cblas_dcopy(m_Nx * m_Ny_loc, fu_loc, 1, k2_u, 1);
    cblas_dcopy(m_Nx * m_Ny_loc, fv_loc, 1, k2_v, 1);
    cblas_dcopy(m_Nx * m_Ny_loc, fh_loc, 1, k2_h, 1);

    // Calculating k3 = f(yn+dt*k2/2) =============================
    // reset temp values
    cblas_dcopy(m_Nx * m_Ny_loc, u_loc, 1, tu_loc, 1); // reset tu to u
    cblas_dcopy(m_Nx * m_Ny_loc, v_loc, 1, tv_loc, 1);
    cblas_dcopy(m_Nx * m_Ny_loc, h_loc, 1, th_loc, 1);

    // update un to un+dt*k2/2 to evaluate f for k3
    cblas_daxpy(m_Nx * m_Ny_loc, m_dt / 2.0, k2_u, 1, tu_loc, 1);
    cblas_daxpy(m_Nx * m_Ny_loc, m_dt / 2.0, k2_v, 1, tv_loc, 1);
    cblas_daxpy(m_Nx * m_Ny_loc, m_dt / 2.0, k2_h, 1, th_loc, 1);

    MPI_Gatherv(tu_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, tu, mpi_info->recvcounts,
                mpi_info->displs, MPI_DOUBLE, 0, mpi_info->m_comm);
    MPI_Gatherv(tv_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, tv, mpi_info->recvcounts,
                mpi_info->displs, MPI_DOUBLE, 0, mpi_info->m_comm);
    MPI_Gatherv(th_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, th, mpi_info->recvcounts,
                mpi_info->displs, MPI_DOUBLE, 0, mpi_info->m_comm);

    Evaluate_f(tu, tv, th, tu_loc, tv_loc, th_loc, fu_loc, fv_loc, fh_loc,
               mpi_info);

    cblas_dcopy(m_Nx * m_Ny_loc, fu_loc, 1, k3_u, 1);
    cblas_dcopy(m_Nx * m_Ny_loc, fv_loc, 1, k3_v, 1);
    cblas_dcopy(m_Nx * m_Ny_loc, fh_loc, 1, k3_h, 1);

    // k4 = f(yn+dt*k3) ===========================================
    // reset temp values
    cblas_dcopy(m_Nx * m_Ny_loc, u_loc, 1, tu_loc, 1); // reset tu to u
    cblas_dcopy(m_Nx * m_Ny_loc, v_loc, 1, tv_loc, 1);
    cblas_dcopy(m_Nx * m_Ny_loc, h_loc, 1, th_loc, 1);

    // update un to un+dt*k2/2 to evaluate f for k3
    cblas_daxpy(m_Nx * m_Ny_loc, m_dt, k3_u, 1, tu_loc, 1);
    cblas_daxpy(m_Nx * m_Ny_loc, m_dt, k3_v, 1, tv_loc, 1);
    cblas_daxpy(m_Nx * m_Ny_loc, m_dt, k3_h, 1, th_loc, 1);

    MPI_Gatherv(tu_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, tu, mpi_info->recvcounts,
                mpi_info->displs, MPI_DOUBLE, 0, mpi_info->m_comm);
    MPI_Gatherv(tv_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, tv, mpi_info->recvcounts,
                mpi_info->displs, MPI_DOUBLE, 0, mpi_info->m_comm);
    MPI_Gatherv(th_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, th, mpi_info->recvcounts,
                mpi_info->displs, MPI_DOUBLE, 0, mpi_info->m_comm);

    Evaluate_f(tu, tv, th, tu_loc, tv_loc, th_loc, fu_loc, fv_loc, fh_loc,
               mpi_info);

    cblas_dcopy(m_Nx * m_Ny_loc, fu_loc, 1, k4_u, 1);
    cblas_dcopy(m_Nx * m_Ny_loc, fv_loc, 1, k4_v, 1);
    cblas_dcopy(m_Nx * m_Ny_loc, fh_loc, 1, k4_h, 1);

    // yn+1 = yn + 1/6*(k1+2*k2+2*k3+k4)*dt
    // Update solution
    for (int i = 0; i < m_Nx; ++i)
    {
        for (int j = 0; j < m_Ny_loc; ++j)
        {
            u_loc[i * m_Ny_loc + j] +=
                m_dt / 6.0 *
                (k1_u[i * m_Ny_loc + j] + 2.0 * k2_u[i * m_Ny_loc + j] +
                 2.0 * k3_u[i * m_Ny_loc + j] + k4_u[i * m_Ny_loc + j]);
            v_loc[i * m_Ny_loc + j] +=
                m_dt / 6.0 *
                (k1_v[i * m_Ny_loc + j] + 2.0 * k2_v[i * m_Ny_loc + j] +
                 2.0 * k3_v[i * m_Ny_loc + j] + k4_v[i * m_Ny_loc + j]);
            h_loc[i * m_Ny_loc + j] +=
                m_dt / 6.0 *
                (k1_h[i * m_Ny_loc + j] + 2.0 * k2_h[i * m_Ny_loc + j] +
                 2.0 * k3_h[i * m_Ny_loc + j] + k4_h[i * m_Ny_loc + j]);
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

    if (mpi_info->m_rank == 0)
    {
        delete[] tu;
        delete[] tv;
        delete[] th;
    }

    delete[] tu_loc;
    delete[] tv_loc;
    delete[] th_loc;
}

void ShallowWater::Solve(int argc, char *argv[])
{

    // ======================================================
    // Create MPI Comminications
    Comm comm;
    Comm::MPI_Info mpi_info;
    m_Ny_loc = comm.CreateMPI(argc, argv, m_Ny, &mpi_info);
    cout << "Ny_loc: " << m_Ny_loc << endl;
    comm.CalcSendParams(m_Nx, &mpi_info);

    // Memory Allocation for local solution fields
    m_u_loc = new double[m_Nx * m_Ny_loc];
    m_v_loc = new double[m_Nx * m_Ny_loc];
    m_h_loc = new double[m_Nx * m_Ny_loc];

    // Memory Allocation for solution fields (root rank)
    if (mpi_info.m_rank == 0)
    {
        m_u = new double[m_Nx * m_Ny];
        m_v = new double[m_Nx * m_Ny];
        m_h = new double[m_Nx * m_Ny];
        m_h0 = new double[m_Nx * m_Ny];
    }

    double *fu_loc = new double[m_Nx * m_Ny_loc];
    double *fv_loc = new double[m_Nx * m_Ny_loc];
    double *fh_loc = new double[m_Nx * m_Ny_loc];

    //  =====================================================
    // Set Initial conditions
    SetInitialConditions(&mpi_info);

    // ======================================================
    // 4th order RK Time Integrations

    // Time advancement
    double time = 0.0; // start time
    while (time <= m_T)
    {
        TimeIntegration(m_u_loc, m_v_loc, m_h_loc, fu_loc, fv_loc, fh_loc,
                        &mpi_info);
        time += m_dt;
    }

    // Gather to global solution fields
    MPI_Gatherv(m_u_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, m_u, mpi_info.recvcounts,
                mpi_info.displs, MPI_DOUBLE, 0, mpi_info.m_comm);
    MPI_Gatherv(m_v_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, m_v, mpi_info.recvcounts,
                mpi_info.displs, MPI_DOUBLE, 0, mpi_info.m_comm);
    MPI_Gatherv(m_h_loc, m_Nx * m_Ny_loc, MPI_DOUBLE, m_h, mpi_info.recvcounts,
                mpi_info.displs, MPI_DOUBLE, 0, mpi_info.m_comm);

    // ======================================================
    // Write to file

    // Write in root rank
    if (mpi_info.m_rank == 0)
    {
        ofstream vOut("output.txt", ios::out | ios ::trunc);
        vOut.precision(5);
        for (int j = 0; j < m_Ny; ++j)
        {
            for (int i = 0; i < m_Nx; ++i)
            {
                vOut << setw(15) << i * m_dx << setw(15) << j * m_dy << setw(15)
                     << m_u[i * m_Ny + j] << setw(15) << m_v[i * m_Ny + j] << setw(15)
                     << m_h[i * m_Ny + j] << endl;
            }
            vOut << endl;
        }

        if (m_method == 'l')
        {
            cout << "Evaluated using loop-based approach" << endl;
        }
        else if (m_method == 'b')
        {
            cout << "Evaluated using BLAS-based approach" << endl;
        }
    }

    // Memory deallocations

    if (mpi_info.m_rank == 0)
    {
        delete[] m_u;
        delete[] m_v;
        delete[] m_h;
        delete[] m_h0;
    }

    delete[] m_u_loc;
    delete[] m_v_loc;
    delete[] m_h_loc;

    delete[] fu_loc;
    delete[] fv_loc;
    delete[] fh_loc;

    comm.DeallocateSendParams(&mpi_info);
    MPI_Finalize();
}

ShallowWater::~ShallowWater() 
{
}