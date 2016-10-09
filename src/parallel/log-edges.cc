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
                	tempX = x + i - 2;
                	tempY = y + j - 2;
                	
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
	
	long start_t, end_t, total_t; /* time measure */
	
	int start, end, startH, endH, sliceH;
	
	int rank; /* rank of process */
	int p;       /* number of processes */
	MPI_Status status;   /* return status for receive */

	/* start up MPI */
	MPI_Init(&argc, &argv);
	
	/* find out process rank */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
	
	/* find out number of processes */
	MPI_Comm_size(MPI_COMM_WORLD, &p);

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
		
		int w = inImg->GetWidth();
		int h = inImg->GetHeight();
		int amount = w * h;
		
		outMat = (int*) malloc(sizeof(int) * amount);
			
		for (int y = 0; y < h; y++) {
			for (int x = 0; x < w; x++) {
				outMat[x + y * h] = inImg->GetGrayValue(x, y);
			}
		}
		
		/*for (int i = 1; i < p; i++) {
			/* find out which part of the image to send */
			/*start = ((int) ((1.0 * amount / p) * i));
			end = ((int) ((1.0 * amount / p) * (i + 1)));
			
			startH = ((int) ((1.0 * h / p) * i));
			endH = ((int) ((1.0 * h / p) * (i + 1)));
			sliceH = endH - startH;
			
			MPI_Send(&w, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			MPI_Send(&sliceH, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
			MPI_Send(outMat + start, end - start, MPI_INT, i, 2, MPI_COMM_WORLD);
		}*/
		
		start = ((int) ((1.0 * amount / p) * rank));
		end = ((int) ((1.0 * amount / p) * (rank + 1)));
		
		startH = ((int) ((1.0 * h / p) * rank));
		endH = ((int) ((1.0 * h / p) * (rank + 1)));
		sliceH = endH - startH;
		
		/* starts timer */
		start_t = get_nanos();
		
		/* applies filter */
		outMat = applyFilter(outMat, w, sliceH);
		
		/* finishes timer */
		end_t = get_nanos();
		
		total_t = end_t - start_t;
		cout << "Time elapsed: " << total_t << "ns" << endl;

		/*for (int y = 0; y < inImg->GetHeight(); y++) {
			for (int x = 0; x < inImg->GetWidth(); x++) {
				outImg->SetGrayValue(x, y, outMat[x + y * inImg->GetHeight()]);
			}
		}*/
		
		outImg->Save("examples/lenaGrayOut.png");
		
		free(inImg);
		free(outImg);
		free(outMat);
	} else {
		/*int w, h;
		
		MPI_Recv(&w, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&h, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
		
		int amount = w * h;
		int *mat = (int*) malloc(sizeof(int) * amount);
		
		MPI_Recv(&mat, amount, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);*/
	}
	
	/* shuts down MPI */
	MPI_Finalize();
	
	return 0;
}
