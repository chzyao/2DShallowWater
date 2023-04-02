#ifndef SHALLOWWATER_H
#define SHALLOWWATER_H

#include "Comm.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

class ShallowWater
{

public:
    ShallowWater();

    void SetParameters(int argc, char *argv[]);

    void Solve(int argc, char *argv[]);

    ~ShallowWater();

protected:

    void SetInitialConditions(Comm::MPI_Info *mpi_info);

    inline int modify_boundary_idx(int idx, int bc_size);

    void SpatialDiscretisation(double *u, double *u_loc, char dir, double *deriv, double *deriv_loc, Comm::MPI_Info *mpi_info);

    void Evaluate_f(double *u, double *v, double *h, double *u_loc, double *v_loc, double *h_loc, double *fu_loc, double *fv_loc, double *fh_loc, Comm::MPI_Info *mpi_info);

    void TimeIntegration(double *u, double *v, double *h, double *fu, double *fv, double *fh, Comm::MPI_Info *mpi_info);

private:
    double m_dt;
    double m_T;
    int m_Nx;
    int m_Ny;
    int m_Ny_loc; // local Ny (for MPI)
    int m_ic;
    double m_dx;
    double m_dy;
    char m_method;

    // Global solution fields
    double *m_u;
    double *m_v;
    double *m_h;
    double *m_h0; // initial condition

    // Local solution fields
    double *m_u_loc;
    double *m_v_loc;
    double *m_h_loc;
};




#endif