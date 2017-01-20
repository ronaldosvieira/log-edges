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
#include <string>
#include <cstdio>
#include <ctime>
#include <cmath>
#include "mpi.h"
#include "pixelLab.h"
#include "logcm.h"

#define DEBUG 1
#define printflush(s, ...) do {if (DEBUG) {printf(s, ##__VA_ARGS__); fflush(stdout);}} while (0)

using std::cout;
using std::endl;
using std::string;

static long get_nanos(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (long) ts.tv_sec * 1000000000L + ts.tv_nsec;
}

bool isInBounds(int x, int y, int w, int h) {
	return x >= 0 && x < w  && y >= 0 && y < h;
}

int* applyFilter(int *orig, int w, int h) {
	int sum, amount, tempX, tempY;
	
	int *mod = (int*) malloc(sizeof(int) * w * h);
	
	/* for each pixel in the image */
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            sum = 0;
            amount = 0;
            
            /* apply the kernel matrix */
            for (int j = 0; j < 5; ++j) {
                for (int i = 0; i < 5; ++i) {
                	tempX = x + i - ((int) (5 / 2));
                	tempY = y + j - ((int) (5 / 2));
                	
                    if (isInBounds(tempX, tempY, w, h)) {
                        sum += lapOfGau[i][j] * orig[tempX + tempY * h];
                        amount += lapOfGau[i][j];
                    }
                }
            }
            
            if (amount) sum /= amount;
            
            if (sum > 255) sum = 255;
            if (sum < 0) sum = 0;
            
            mod[x + y * h] = sum;
        }
    }
    
    return mod;
}

int main(int argc, char* argv[]) {
	PixelLab *inImg = new PixelLab(); /* input image */
	PixelLab *outImg = new PixelLab(); /* output image */
	
	int *outMat;
	
	double start_t, end_t, total_t; /* time measure */
	
	int slice, width, height, amount, *mat;
	
	int rank; /* rank of process */
	int p; /* number of processes */
	MPI_Status status; /* return status for receive */

	/* start up MPI */
	MPI_Init(&argc, &argv);
	
	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);

	/* validates arguments */
	if (argc != 2) {
		if (rank == 0)
			cout << "Usage: " << argv[0] << " (image path)" << endl;

		MPI_Finalize();
		return -1;
	}

	/* pre processing */
	if (rank == 0) {
		string inImgPath = argv[1];
		FILE *fp = fopen(inImgPath.c_str(), "rb");
	
		if (!fp) {
			if (rank == 0)
				cout << "Error: image '" << inImgPath << "' not found." << endl;
	
			MPI_Finalize();
			return -1;
		}
		
		fclose(fp);
		
		inImg->Read(inImgPath.c_str());
		outImg->Copy(inImg);
		
		width = inImg->GetWidth();
		height = inImg->GetHeight();
		amount = width * height;
		slice = amount / p;
		
		cout << "w = " << width << "; h = " << height << endl;
		cout << "amount = " << amount << "; slice = " << slice << endl;
		
		/* starts timer */
		start_t = MPI_Wtime();
		
		outMat = (int*) malloc(sizeof(int) * amount);
			
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				outMat[x + y * height] = inImg->GetGrayValue(x, y);
			}
		}
	}
	
	/* splits image */
	if (rank == 0) {
		for (int i = 1; i < p; i++) {
			MPI_Send(&slice, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&width, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
			
			MPI_Send(outMat + (slice * i), slice, MPI_INT, i, 2, MPI_COMM_WORLD);
		}
		
		mat = outMat;
	} else {
		MPI_Recv(&slice, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&width, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		
		mat = (int*) malloc(sizeof(int) * slice);
		
		MPI_Recv(mat, slice, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
	}
	
	/* applies filter */
	mat = applyFilter(mat, width, slice / width);
	
	/* joins image */
	if (rank == 0) {
		// copy its own part into result matrix
		memcpy(outMat, mat, sizeof(int) * slice);
		
		int* temp = (int*) malloc(sizeof(int) * slice);
		
		for (int i = 1; i < p; i++) {
			MPI_Recv(temp, slice, MPI_INT, MPI_ANY_TAG, 3, MPI_COMM_WORLD, &status);
			
			memcpy(outMat + (slice * status.MPI_SOURCE * 0), temp, sizeof(int) * slice);
		}
		
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				outImg->SetGrayValue(x, y, outMat[x + y * inImg->GetHeight()]);
			}
		}
		
		// finishes timer
		end_t = MPI_Wtime();
		
		total_t = end_t - start_t;
		cout << "Time elapsed: " << total_t << "s" << endl;
		
		outImg->Save("examples/lenaGrayOut.png");
		
		free(temp);
		free(inImg);
		free(outImg);
		free(outMat);
	} else {
		MPI_Send(mat, slice, MPI_INT, 0, 3, MPI_COMM_WORLD);
		
		free(mat);
	}
	
	// shuts down MPI
	MPI_Finalize();
	
	return 0;
}