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

    // Domain Decompositions
    // Handling arbitrary number of processes
    int remainder = Ny % mpi_info->m_size;
    int chunk_size = (Ny - remainder) / mpi_info->m_size;

    if (mpi_info->m_rank < remainder)
    {
        m_Ny_loc = chunk_size + 1;
    }
    else
    {
        m_Ny_loc = chunk_size;
    }

    // Store Ny_loc of each rank to an array for later use in Scatterv and Gatherv
    mpi_info->m_Ny_loc_array = new int[mpi_info->m_size];
    mpi_info->m_Ny_loc_array[mpi_info->m_rank] = m_Ny_loc;

    // Make sure that every rank has m_Ny_loc_array for calcuting send params in later stages
    MPI_Allgather(&m_Ny_loc, 1, MPI_INT, mpi_info->m_Ny_loc_array, 1, MPI_INT, MPI_COMM_WORLD);


    return m_Ny_loc;
}

void Comm::CalcSendParams(int Nx, MPI_Info *mpi_info)
{
    mpi_info->sendcounts = nullptr;
    mpi_info->recvcounts = nullptr;
    mpi_info->displs = nullptr;

    if (mpi_info->m_rank == 0)
    {
        // Array of size of send buffer in each rank
        mpi_info->sendcounts = new int[mpi_info->m_size];
        mpi_info->recvcounts = new int[mpi_info->m_size];
        for (int i = 0; i < mpi_info->m_size; ++i)
        {
            mpi_info->sendcounts[i] = Nx * mpi_info->m_Ny_loc_array[i];
            mpi_info->recvcounts[i] = Nx * mpi_info->m_Ny_loc_array[i];
            cout << "rank: " << i << ",  send count" << mpi_info->sendcounts[i] << endl;
        }

        // Array of offset displacement of send buffer in each rank
        mpi_info->displs = new int[mpi_info->m_size];
        mpi_info->displs[0] = 0;
        for (int i = 1; i < mpi_info->m_size; ++i)
        {
            mpi_info->displs[i] = mpi_info->displs[i - 1] + mpi_info->sendcounts[i - 1];
            cout << "rank: " << i << ",  displs" << mpi_info->displs[i] << endl;
        }
    }
}

void Comm::DeallocateSendParams(MPI_Info *mpi_info)
{
    delete[] mpi_info->sendcounts;
    delete[] mpi_info->displs;
}

Comm::~Comm()
{
}
