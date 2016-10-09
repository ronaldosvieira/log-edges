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
	
	pixel **inMat;
	int *outMat;
	
	long start_t, end_t, total_t; /* time measure */
	
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
		
		inImg->AllocatePixelMatrix(&inMat, inImg->GetHeight(), inImg->GetWidth());
		inImg->GetDataAsMatrix(inMat);
		
		outMat = (int*) 
			malloc(sizeof(int) * inImg->GetHeight() * inImg->GetWidth());
			
		for (int y = 0; y < inImg->GetHeight(); y++) {
			for (int x = 0; x < inImg->GetWidth(); x++) {
				outMat[x + y * inImg->GetHeight()] = 
					inImg->GetGrayValue(x, y);
			}
		}
		
		/*for (int i = 0; i < p; i++) {
			/* find out which part of the image to send */
			/*int start = ((int) ((1.0 * outMat->GetHeight() / p) * my_rank));
			int end = ((int) ((1.0 * GetHeight() / p) * (my_rank + 1)));
			
			MPI_Send(outMat + start, end - start, MPI_INT, ,);
		}*/
		
		/* starts timer */
		start_t = get_nanos();
		
		/* applies filter */
		outMat = applyFilter(outMat, inImg->GetWidth(), inImg->GetHeight());
		
		/* finishes timer */
		end_t = get_nanos();
		
		total_t = end_t - start_t;
		cout << "Time elapsed: " << total_t << "ns" << endl;

		for (int y = 0; y < inImg->GetHeight(); y++) {
			for (int x = 0; x < inImg->GetWidth(); x++) {
				outImg->SetGrayValue(x, y, outMat[x + y * inImg->GetHeight()]);
			}
		}
		
		inImg->DeallocatePixelMatrix(&inMat, inImg->GetHeight(), inImg->GetWidth());
		
		outImg->Save("examples/lenaGrayOut.png");
		
		free(inImg);
		free(outImg);
		free(outMat);
	} else {
		/* find out which part of the image to send */
		/*int start = ((int) ((1.0 * outMat->GetHeight() / p) * my_rank));
		int end = ((int) ((1.0 * GetHeight() / p) * (my_rank + 1)));*/
		
		
	}
	
	/* shuts down MPI */
	MPI_Finalize();
	
	return 0;
}
