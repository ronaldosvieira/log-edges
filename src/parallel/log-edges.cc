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
	return x >= 0 && x < w && y >= 0 && y < h;
}

int* applyFilter(int *mat, int w, int h) {
	int sum, amount, tempX, tempY;
	
	int *orig = (int*) malloc(sizeof(int) * w * h);
	memcpy(orig, mat, sizeof(int) * w * h);
	
	/* for each pixel in the image */
    for (int x = 0; x < w; ++x) {
        for (int y = 0; y < h; ++y) {
            sum = 0;
            amount = 0;
            
            /* apply the kernel matrix */
            for (int j = 0; j < 5; ++j) {
                for (int i = 0; i < 5; ++i) {
                	tempX = x + i - ((int) (5 / 2));
                	tempY = y + j - ((int) (5 / 2));
                	
                    if (isInBounds(tempX, tempY, w, h)) {
                        sum += lapOfGau[i][j] * orig[tempX + tempY * w];
                        amount += lapOfGau[i][j];
                    }
                }
            }
            
            if (amount) sum /= amount;
            
            if (sum > 255) sum = 255;
            if (sum < 0) sum = 0;
            
            mat[x + y * w] = sum;
        }
    }
    
    free(orig);
}

int main(int argc, char* argv[]) {
	PixelLab *inImg = new PixelLab(); /* input image */
	PixelLab *outImg = new PixelLab(); /* output image */
	
	int *outMat;
	
	double start_t, end_t, total_t; /* time measure */
	
	int origWidth, origHeight, /* original image size */
		width, height, /* slice size */
		*mat;
		
	int filterOffset = 2;
	int startOffsetY, endOffsetY;
	
	int rank; /* rank of process */
	int p; /* number of processes */
	MPI_Status status; /* return status for receive */

	/* start up MPI */
	MPI_Init(&argc, &argv);
	
	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	/* round down p to nearest power of 2 */
	p = powf(2.0f, floorf(log2f(p)));
	
	/* kill unused processes */
	if (rank >= p) {
		MPI_Finalize();
		return 0;
	}

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
		
		origWidth = inImg->GetWidth();
		origHeight = inImg->GetHeight();
		
		width = origWidth;
		height = origHeight / p;
		
		cout << "w = " << width << "; h = " << height << endl;
		
		/* starts timer */
		start_t = MPI_Wtime();
		
		outMat = (int*) malloc(sizeof(int) * origWidth * origWidth);
			
		for (int y = 0; y < origHeight; y++) {
			for (int x = 0; x < origWidth; x++) {
				outMat[x + y * origWidth] = inImg->GetGrayValue(x, y);
			}
		}
	}
	
	/* splits image */
	if (rank == 0) {
		for (int i = 1; i < p; i++) {
			startOffsetY = height * i - filterOffset < 0? 
				0 : filterOffset;
			endOffsetY = height * (i + 1) + filterOffset > origHeight? 
				0 : filterOffset;
			
			MPI_Send(&width, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&height, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
			
			MPI_Send(&startOffsetY, 1, MPI_INT, i, 8, MPI_COMM_WORLD);
			MPI_Send(&endOffsetY, 1, MPI_INT, i, 9, MPI_COMM_WORLD);
			
			MPI_Send(outMat + (width * height * i) - (startOffsetY * width), 
				width * (startOffsetY + height + endOffsetY),
				MPI_INT, i, 2, MPI_COMM_WORLD);
		}
		
		startOffsetY = 0;
		endOffsetY = filterOffset;
		
		mat = outMat;
	} else {
		MPI_Recv(&width, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&height, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		
		MPI_Recv(&startOffsetY, 1, MPI_INT, 0, 8, MPI_COMM_WORLD, &status);
		MPI_Recv(&endOffsetY, 1, MPI_INT, 0, 9, MPI_COMM_WORLD, &status);
		
		mat = (int*) malloc(sizeof(int) * width * (startOffsetY + height + endOffsetY));
		
		MPI_Recv(mat, width * (startOffsetY + height + endOffsetY), 
			MPI_INT, 0, 2, MPI_COMM_WORLD, &status);
	}
	
	/* applies filter */
	applyFilter(mat, width, startOffsetY + height + endOffsetY);
	
	/* joins image */
	if (rank == 0) {
		int* temp = (int*) malloc(sizeof(int) * width * height);
		
		for (int i = 1; i < p; i++) {
			MPI_Recv(temp, width * height, MPI_INT, 
				MPI_ANY_TAG, 3, MPI_COMM_WORLD, &status);
			
			memcpy(outMat + (width * height * status.MPI_SOURCE), 
				temp, sizeof(int) * width * height);
		}
		
		for (int y = 0; y < origHeight; y++) {
			for (int x = 0; x < origWidth; x++) {
				outImg->SetGrayValue(x, y, outMat[x + y * origWidth]);
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
		MPI_Send(mat + (startOffsetY * width), width * height, 
			MPI_INT, 0, 3, MPI_COMM_WORLD);
		
		free(mat);
	}
	
	// shuts down MPI
	MPI_Finalize();
	
	return 0;
}