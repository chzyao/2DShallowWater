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

    void SpatialDiscretisation(double *u, char dir, double *deriv);

    void Evaluate_f(double *u, double *v, double *h, double *fu, double *fv, double *fh);

    void TimeIntegration(double *u, double *v, double *h, double *fu, double *fv, double *fh);

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

    // Solutions
    double *m_u;
    double *m_v;
    double *m_h;
    double *m_h0; // initial condition

    double *m_u_loc;
    double *m_v_loc;
    double *m_h_loc;
};




#endif