#ifndef COMM_H
#define COMM_H

#include <mpi.h>

class Comm
{
public:
    Comm();

    typedef struct MPI_Info
    {
        MPI_Comm m_comm;
        int m_rank;
        int m_size;
        int m_Ny_loc;
    } MPI_Info;

    int CreateMPI(int argc, char *argv[], int Ny, MPI_Info *mpi_info);

    ~Comm();

private:
    int m_Ny_loc; // local Ny
};

#endif