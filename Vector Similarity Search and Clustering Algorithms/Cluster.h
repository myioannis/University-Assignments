#ifndef CLUSTER_H
#define CLUSTER_H

#include <vector>
#include "Utils.h"
#include "Technique.h"
#include "LSH.h"

using namespace std;

class Image;
class Hypercube;
class Exact;

class Cluster{
    private:
        //Members
        int K; //Number of clusters
        int w; //window
        int L; //Number of hashtables
        int k; //Number of buckets
        int d; //Dimensions/Number of pixels
        int M;
        int probes;
        double r; //Range for range_search
        string method;
        vector<Image*>& nonCentroids;
        vector<Image*> Centroids;
        vector<Image*> newCentroids; //all the <extra> points allocated in the heap, due to the Median
        vector<vector<Image*>> Clusters;
        Distance_Function distance_Function;
        //Methods
        vector<vector<Image*>> assignment(vector<distancePair*>& closestCentroids);
        vector<vector<Image*>> Approximate_assignment(vector<vector<distancePair*>>& closestNeighbors, unordered_map<Image*, pair<int, double>>& assigned, int& numOfChanges);
        void new_Centroids(vector<vector<Image*>> clusters);
        Image* find_newCentroid(vector<Image*> cluster, int cluster_num);
        unsigned char median_ofDimension(vector<unsigned char>& kati);
        bool converged(vector<distancePair*> closestCentroids,vector<distancePair*> prevClosestCentroids);
        bool Approximate_converged(unordered_map<Image*, pair<int, double>>& assigned, int& oldUnassigned, int iteration, int numOfChanges);
        double initialize_Radius();
    public:
        Cluster(int arg_K, int arg_w, int arg_L, int arg_k, int arg_d, int arg_M, int arg_probes, int arg_r, string arg_method, vector<Image*>& arg_nC, Distance_Function arg_distance_Function);
        ~Cluster();
        vector<vector<Image*>> make_Clusters();
        vector<double> compute_Silhouette();
};

#endif