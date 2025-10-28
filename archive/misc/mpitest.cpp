/** This is a simple C++ test to check if cuda-aware MPI works as
 *  expected.
 *  It allocates a GPU array and puts 1s into it, then sends it
 *  across MPI to the receiving rank, which transfers back to 
 *  host and outputs the values. 
 *  The expected output is:
 * 
 *  Received 1, 1
 * 
 * Compile with:
 *   mpic++ -o test mpitest.cpp -L/path/to/cuda/libs -lcudart
 * 
 * Run with:
 *   mpirun -np 2 test
 */

#include <cstdio>
#include <string>
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <iostream>

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int* d_send;
        cudaMalloc((void**)&d_send, 2*sizeof(int));
        int h_send[] = {1, 1};
        cudaMemcpy(d_send, h_send, 2*sizeof(int), cudaMemcpyHostToDevice);
        MPI_Send(d_send, 2, MPI_INT, 1, 99, MPI_COMM_WORLD);
        std::cout << "Data has been sent...\n";
    } else if (rank == 1) {
        int* d_recv;
        cudaMalloc((void**)&d_recv, 2*sizeof(int));
        MPI_Recv(d_recv, 2, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);
        int h_recv[2];
        cudaMemcpy(h_recv, d_recv, 2*sizeof(int), cudaMemcpyDeviceToHost);
        std::cout << "Received " << h_recv[0] << ", " << h_recv[1] << "\n";
    }

}