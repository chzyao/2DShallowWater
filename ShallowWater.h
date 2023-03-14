#ifndef SHALLOWWATER_H
#define SHALLOWWATER_H

#include <boost/program_options.hpp>

namespace po = boost::program_options;

class ShallowWater
{

public:
    ShallowWater(const po::variables_map &vm);

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

    ~ShallowWater();

private:
    // solutions
    double *u;
    double *v;
    double *h;
};

#endif