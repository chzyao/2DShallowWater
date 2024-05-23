/**
 * @file Comm.h
 * @author Chris (chris.yao20@imperial.ac.uk)
 * @version Initial version uploaded in March 2023. Last revised in May 2024.
 */

#ifndef COMM_H
#define COMM_H

#include <mpi.h>

class Comm
{
public:
    Comm();

    typedef struct MPI_Info
    {
        // General MPI Info
        MPI_Comm m_comm;
        int m_rank;
        int m_size;
        int *m_Ny_loc_array;

        // Send parameters for Scatterv and Gatherv
        int *sendcounts; // array of sizes of each send buffer
        int *recvcounts; // array of sizes of each receive buffer
        int *displs;     // array of offsets of each send buffer

    } MPI_Info;

    int CreateMPI(int argc, char *argv[], int Ny, MPI_Info *mpi_info);

    void CalcSendParams(int Nx, MPI_Info *mpi_info);

    void DeallocateSendParams(MPI_Info *mpi_info);

    ~Comm();

private:
    int m_Ny_loc; // local Ny
};

#endif