#ifndef UTILS_H
#define UTILS_H

#include <string>

double randDouble();
void generateCentroids(int Nc, int D, double* initial_centroids);
void generateInitialData(int N, int Nc, int D, double* data, double* initial_centroids, double stddev, unsigned int seed);
void writeData(const std::string& filename, double* data, double* clusterCentroids, int* N_p, int* D_p, int* K_p, double* epsilon_p);
void readData(const std::string& filename, double*& data, double*& clusterCentroids, int* N_p, int* D_p, int* K_p, double* epsilon_p);

#endif
