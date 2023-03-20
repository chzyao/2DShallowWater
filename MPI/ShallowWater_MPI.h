#ifndef SHALLOWWATER_H
#define SHALLOWWATER_H

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

    void SetInitialConditions(double *u, double *v, double *h);

    void SpatialDiscretisation(double *u, char dir, double *deriv);

    void Evaluate_fu(double *u, double *v, double *h, double *f);
    void Evaluate_fv(double *u, double *v, double *h, double *f);
    void Evaluate_fh(double *u, double *v, double *h, double *f);

    // void Evaluate_fu_BLAS(double *u, double *v, double *h, int Nx, int Ny,
    //                       double dx, double dy, double *f);
    // void Evaluate_fv_BLAS(double *u, double *v, double *h, int Nx, int Ny,
    //                       double dx, double dy, double *f);
    // void Evaluate_fh_BLAS(double *u, double *v, double *h, int Nx, int Ny,
    //                       double dx, double dy, double *f);

    void TimeIntegration(double *u, double *v, double *h, double *fu, double *fv, double *fh);

private:
    double m_dt;
    double m_T;
    int m_Nx;
    int m_Ny;
    int m_ic;
    double m_dx;
    double m_dy;

    // Solutions
    double *m_u;
    double *m_v;
    double *m_h;
    double *m_h0; // initial condition
};




#endif