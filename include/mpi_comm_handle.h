#ifndef _COMM_HANDLE_H
#define _COMM_HANDLE_H

#if defined(MPI_COMMS)
#include <mpi.h>
extern MPI_Comm MPI_COMM_HANDLE;
#endif

#endif /* _COMM_HANDLE_H */
