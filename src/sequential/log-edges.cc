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
#include <cstring>
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
using std::min;
using std::max;
using std::string;
using std::memcpy;

static long get_nanos(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (long) ts.tv_sec * 1000000000L + ts.tv_nsec;
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
                	
                	tempX = min(max(tempX, 0), w - 1);
                	tempY = min(max(tempY, 0), h - 1);
                	
                    sum += lapOfGau[i][j] * orig[tempX + tempY * w];
                    amount += lapOfGau[i][j];
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

	/* validates arguments */
	if (argc != 2) {
		cout << "Usage: " << argv[0] << " (image path)" << endl;

		return -1;
	}

	/* pre processing */
	string inImgPath = argv[1];
	FILE *fp = fopen(inImgPath.c_str(), "rb");

	if (!fp) {
		cout << "Error: image '" << inImgPath << "' not found." << endl;

		return -1;
	}
	
	fclose(fp);
	
	inImg->Read(inImgPath.c_str());
	outImg->Copy(inImg);
	
	origWidth = inImg->GetWidth();
	origHeight = inImg->GetHeight();
	
	width = origWidth;
	height = origHeight;
	
	/* starts timer */
	start_t = MPI_Wtime();
	
	outMat = (int*) malloc(sizeof(int) * origWidth * origWidth);
		
	for (int y = 0; y < origHeight; y++) {
		for (int x = 0; x < origWidth; x++) {
			outMat[x + y * origWidth] = inImg->GetGrayValue(x, y);
		}
	}
	
	/* applies filter */
	applyFilter(outMat, width, height);
		
	// finishes timer
	end_t = MPI_Wtime();
	
	total_t = end_t - start_t;
	cout << "Time elapsed: " << total_t << "s" << endl;
	
	outImg->Save("examples/lenaGrayOut.png");
	
	free(inImg);
	free(outImg);
	free(outMat);
	
	return 0;
}