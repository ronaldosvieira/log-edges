/*
 ============================================================================
 Name        : log-edges.cc
 Author      : Ronaldo Vieira
 Version     : 0.0.1
 Copyright   : MIT License
 Description : Edge detection in images using a parallel and distributed 
laplacian-of-gaussian filter.
 ============================================================================
 */
#include <iostream>
#include <cstdio>
#include <ctime>
#include <cmath>
#include "mpi.h"

#define DEBUG 0
#define printflush(s, ...) do {if (DEBUG) {printf(s, ##__VA_ARGS__); fflush(stdout);}} while (0)

static long get_nanos(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (long) ts.tv_sec * 1000000000L + ts.tv_nsec;
}

int* get_random_nums(int amount) {
    int *rand_nums = (int *) malloc(sizeof(int) * amount);
    int i;

    for (i = 0; i < amount; i++) {
        rand_nums[i] = rand() % 10;
    }

    return rand_nums;
}

int recSum(int *rand_nums, int amount, int p, int my_rank, int parent_rank) {
	printflush("%d: sumRec with %d-sized array and p = %d\n", my_rank, amount, p);
	int sum, child_rank, new_amount, new_p;

	if (p == 1 /*|| amount <= 2*/) {
		int i;
		sum = 0;

		for (i = 0; i < amount; i++) {
			sum += rand_nums[i];
		}

		printflush("%d: Partial sum: %d\n", my_rank, sum);
	} else {
		child_rank = std::ceil(my_rank + (p / 2.0));
		new_amount = (int) amount / 2;
		new_p = (int) p / 2;

		MPI_Send(&new_amount, 1, MPI_INT, child_rank, 2, MPI_COMM_WORLD);
		MPI_Send(&new_p, 1, MPI_INT, child_rank, 3, MPI_COMM_WORLD);
		MPI_Send(rand_nums + (amount - new_amount), new_amount, MPI_INT, child_rank, 0, MPI_COMM_WORLD);

		printflush("%d: Sent to %d\n", my_rank, child_rank);

		int sum1 = recSum(rand_nums, amount - new_amount, p - new_p, my_rank, my_rank);
		int sum2;
		MPI_Status status;

		MPI_Recv(&sum2, 1, MPI_INT, child_rank, 1, MPI_COMM_WORLD, &status);

		sum = sum1 + sum2;
	}

	if (parent_rank == my_rank) {
		return sum;
	} else {
		MPI_Send(&sum, 1, MPI_INT, parent_rank, 1, MPI_COMM_WORLD);
		printflush("%d: Sent sum to %d\n", my_rank, parent_rank);

		return -1;
	}
}

int main(int argc, char* argv[]){
	srand(time(NULL));

	int i;
	int sum;
	int *rand_nums = NULL; /* array of random numbers */
	int my_rank; /* rank of process */
	int p;       /* number of processes */
	MPI_Status status;   /* return status for receive */
	
	long start_t, end_t, total_t; /* time measure */

	/* start up MPI */
	
	MPI_Init(&argc, &argv);
	
	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
	
	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	if (argc != 2) {
		if (my_rank == 0)
			printf("Usage: %s (size of array >= 2*p)\n", argv[0]);

		MPI_Finalize();
		return -1;
	}

	int amount = atoi(argv[1]);

	if (amount < 0) {
		if (my_rank == 0)
			printf("Error: Size of array must be greater than zero.\n");

		MPI_Finalize();
		return -1;
	}

	/* rank 0 creates and sends the array */
	if (my_rank == 0) {
		rand_nums = (int*) get_random_nums(amount);

		printflush("%d: Created the array: [", my_rank);
		for (i = 0; i < amount - 1; i++) {
			printflush("%d, ", rand_nums[i]);
		}
		printflush("%d]\n", rand_nums[amount - 1]);
		
		start_t = get_nanos();

		sum = recSum(rand_nums, amount, p, my_rank, my_rank);

		end_t = get_nanos();
		total_t = end_t - start_t;

        printflush("%d: End. Sum is %d\n", my_rank, sum);
        
        printf("%d: Time elapsed: %ldns\n", my_rank, total_t);
	/* other ranks receive the array from rank 0 */
	} else {
		MPI_Recv(&amount, 1, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &status);
		rand_nums = (int *) malloc(sizeof(int) * amount);

		MPI_Recv(&p, 1, MPI_INT, status.MPI_SOURCE, 3, MPI_COMM_WORLD, &status);
		MPI_Recv(rand_nums, amount, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD, &status);
		printflush("%d: Received %d-sized array from %d \n", my_rank, amount, status.MPI_SOURCE);

		recSum(rand_nums, amount, p, my_rank, status.MPI_SOURCE);
	}

	free(rand_nums);

	/* shut down MPI */
	MPI_Finalize(); 
	
	return 0;
}
