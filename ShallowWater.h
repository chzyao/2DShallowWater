#ifndef SHALLOWWATER_H
#define SHALLOWWATER_H

class ShallowWater
{

public:
    ShallowWater(double dt, double T, int Nx, int Ny, int ic);
    ~ShallowWater();
    void SetInitialConditions();
    void TimeIntegrate();


private:
    // solutions
    double u;
    double v;
    double h;

};

ShallowWater::ShallowWater(double dt, double T, int Nx, int Ny, int ic)
{
}

ShallowWater::~ShallowWater()
{
}







#endif