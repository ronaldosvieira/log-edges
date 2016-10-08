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

bool isInBounds(int x, int y, int w, int h) {
	return x >= 0 && x < w  && y >= 0 && y < h;
}

static long get_nanos(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (long) ts.tv_sec * 1000000000L + ts.tv_nsec;
}

void applyFilter(pixel **inMat, pixel **outMat, int w, int h) {
	int sum, amount;
	
	/* for each pixel in the image */
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            sum = 0;
            amount = 0;
            
            /* apply the kernel matrix */
            for (int j = 0; j < 5; ++j) {
                for (int i = 0; i < 5; ++i) {
                    if (isInBounds(x + i - 2, y + j - 2, w, h)) {
                        sum += lapOfGau[i][j] * inMat[y + j - 2][x + i - 2].value;
                        amount += lapOfGau[i][j];
                    }
                }
            }
            
            if (amount) sum /= amount;
            
            if (sum > 255) sum = 255;
            if (sum < 0) sum = 0;
            
            outMat[y][x].value = sum;
        }
    }
}

int main(int argc, char* argv[]) {
	PixelLab *inImg = new PixelLab(); /* input image */
	PixelLab *outImg = new PixelLab(); /* output image */
	
	pixel **inMat, **outMat;
	
	long start_t, end_t, total_t; /* time measure */

	if (argc != 2) {
		cout << "Usage: " << argv[0] << " (image path)" << endl;

		return -1;
	}

	string inImgPath = argv[1];
	FILE *fp = fopen(inImgPath.c_str(), "rb");

	if (!fp) {
		cout << "Error: image '" << inImgPath << "' not found." << endl;

		return -1;
	}
	
	fclose(fp);
	
	inImg->Read(inImgPath.c_str());
	outImg->Copy(inImg);
	
	inImg->AllocatePixelMatrix(&inMat, inImg->GetHeight(), inImg->GetWidth());
	inImg->GetDataAsMatrix(inMat);
	outImg->AllocatePixelMatrix(&outMat, outImg->GetHeight(), outImg->GetWidth());
	outImg->GetDataAsMatrix(outMat);
	
	/* starts timer */
	start_t = get_nanos();
	
	/* applies filter */
	applyFilter(inMat, outMat, inImg->GetWidth(), inImg->GetHeight());
    
    /* finishes timer */
    end_t = get_nanos();
    
    total_t = end_t - start_t;
    cout << "Time elapsed: " << total_t << "ns" << endl;
	
	outImg->SetDataAsMatrix(outMat);
	
	inImg->DeallocatePixelMatrix(&inMat, inImg->GetHeight(), inImg->GetWidth());
	outImg->DeallocatePixelMatrix(&outMat, outImg->GetHeight(), outImg->GetWidth());
	
	outImg->Save("examples/lenaGrayOut.png");
	
	free(inImg);
	free(outImg);
	
	return 0;
}
