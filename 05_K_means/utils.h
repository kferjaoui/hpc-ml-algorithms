#ifndef UTILS_H
#define UTILS_H

double randDouble();
void generateInitialCentroids(int Nc, int D, double* initial_centroids);
void generateInitialData(int N, int Nc, int D, double* data, double* initial_centroids, double stddev, unsigned int seed);

#endif
