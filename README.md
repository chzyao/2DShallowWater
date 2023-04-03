# 2DShallowWater
HPC coursework 2023

This code solves the simplified 2D Shallow Water Equation using MPI and OpenMP.

To compile the code, it is necessary to use Cmake as the default command line input parameters were set there. For the best performance, it is recommended to use 
Loop: np = 2, OMP_NUM_THREADS=2
BLAS: np = 4, OMP_NUM_THREADS=4

output.txt will be generated in the build directory
