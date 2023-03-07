#ifndef SHALLOWWATER_H
#define SHALLOWWATER_H

class ShallowWater
{

public:
    ShallowWater(); // default constructor
    ShallowWater(double dt, double T, int Nx, int Ny, int ic);
    ~ShallowWater();
    void SetInitialConditions(int ic);
    void TimeIntegrate();


private:
    // solutions
    double u;
    double v;
    double h;
    double t;


};









#endif