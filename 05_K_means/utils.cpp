#include <cstdlib>
// #include <experimental/random>
#include <random>
#include "utils.h"


// Generates random double precision number in [0,1]
double randDouble(){
    return static_cast<double>(std::rand()) / RAND_MAX;
}

// Randomly generates Nc centroids with coordinates between [0,1]
void generateInitialCentroids(int Nc, int D, double* initial_centroids){
    for(int i=0; i<Nc; i++){
        for(int j=0; j<D; j++){
            initial_centroids[i*D+j] = randDouble(); 
        }
    }
}

// Randomly generates N data points around Nc centroids
void generateInitialData(int N, int Nc, int D, double* data, double* initial_centroids, double stddev, unsigned int seed){
    std::mt19937 generator(seed);
    std::normal_distribution<double> noise{0.0, stddev};   
    std::uniform_int_distribution<> dist(0, Nc-1);

    for(int i=0; i<N; i++){
        int random_centroid_id = dist(generator); //std::experimental::randint(0,Nc-1);
        for(int k=0; k<D; k++){
            data[i*D+k] = initial_centroids[random_centroid_id*D+k] + noise(generator);
        }
    }
}
