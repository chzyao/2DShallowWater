#ifndef SHALLOWWATER_H
#define SHALLOWWATER_H

#include <boost/program_options.hpp>

namespace po = boost::program_options;

class ShallowWater
{

public:

    ShallowWater();

    ShallowWater(double dt, double T, int Nx, int Ny, int ic);

    static void SetParameters(int argc, char *argv[]);

    void SetInitialConditions(double *u, double *v, double *h, double *h0,
                              int Nx, int Ny, int ic, double dx, double dy);

    void SpatialDiscretisation(double *u, int Nx, int Ny, double dx, double dy,
                               char dir, double *deriv);

    void Evaluate_fu(double *u, double *v, double *h, int Nx, int Ny,
                     double dx, double dy, double *f);
    void Evaluate_fv(double *u, double *v, double *h, int Nx, int Ny,
                     double dx, double dy, double *f);
    void Evaluate_fh(double *u, double *v, double *h, int Nx, int Ny,
                     double dx, double dy, double *f);

    void Evaluate_fu_BLAS(double *u, double *v, double *h, int Nx, int Ny,
                         double dx, double dy, double *f);
    void Evaluate_fv_BLAS(double *u, double *v, double *h, int Nx, int Ny,
                         double dx, double dy, double *f);
    void Evaluate_fh_BLAS(double *u, double *v, double *h, int Nx, int Ny, 
                         double dx, double dy, double *f);

    void TimeIntegration(double *u, double *v, double *h, int Nx, int Ny,
                         double dx, double dy, double dt, double *fu,
                         double *fv, double *fh);

    // temp test function
    void printMatrix(int Nx, int Ny, double *A);

    void Solve();

    ~ShallowWater();

private:

    // Input parameters
    double *u;
    double *v;
    double *h;
    const double dt;
    const double T;
    const int Nx;
    const int Ny;
    const int ic;

};

#endif