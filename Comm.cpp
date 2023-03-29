#include "Comm.h"
#include <iostream>

using namespace std;

Comm::Comm()
{
}

int Comm::CreateMPI(int argc, char *argv[], int Ny, MPI_Info *mpi_info)
{
    MPI_Init(&argc, &argv);
    mpi_info->m_comm = MPI_COMM_WORLD;

    // Get size and rank
    int retval_rank, retval_size;
    retval_rank = MPI_Comm_rank(MPI_COMM_WORLD, &mpi_info->m_rank);
    retval_size = MPI_Comm_size(MPI_COMM_WORLD, &mpi_info->m_size);

    if (retval_rank == MPI_ERR_COMM || retval_size == MPI_ERR_COMM)
    {
        cout << "Invalid communicator" << endl;
        return 1;
    }

    m_Ny_loc = Ny / mpi_info->m_size; 
    return m_Ny_loc;
}



Comm::~Comm()
{
}
