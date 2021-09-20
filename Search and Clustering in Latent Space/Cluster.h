#ifndef CLUSTER_H
#define CLUSTER_H

#include <vector>
#include "Utils.h"
#include "Technique.h"
#include "LSH.h"

using namespace std;

template <typename T>
class Image;
template <typename T>
class Hypercube;
template <typename T>
class Exact;

template <typename T>
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
        vector<Image<T>*>& nonCentroids;
        vector<Image<T>*> Centroids;
        vector<Image<T>*> newCentroids; //all the <extra> points allocated in the heap, due to the Median
        vector<vector<Image<T>*>> Clusters;
        Distance_Function<T> distance_Function;
        //Methods
        vector<vector<Image<T>*>> assignment(vector<distancePair<T>*>& closestCentroids);
        vector<vector<Image<T>*>> Approximate_assignment(vector<vector<distancePair<T>*>>& closestNeighbors, unordered_map<Image<T>*, pair<int, double>>& assigned, int& numOfChanges);
        T median_ofDimension(vector<T>& kati);
        unsigned char median_ofDimension_Reduced(vector<unsigned char>& allValuesOfDimension);
        bool converged(vector<distancePair<T>*> closestCentroids,vector<distancePair<T>*> prevClosestCentroids);
        bool Approximate_converged(unordered_map<Image<T>*, pair<int, double>>& assigned, int& oldUnassigned, int iteration, int numOfChanges);
        double initialize_Radius();
    public:
        Cluster(int arg_K, vector<Image<T>*>& arg_nC, Distance_Function<T> arg_distance_Function);
        Cluster(int arg_K, int arg_w, int arg_L, int arg_k, int arg_d, int arg_M, int arg_probes, int arg_r, string arg_method, vector<Image<T>*>& arg_nC, Distance_Function<T> arg_distance_Function);
        ~Cluster();
        void new_Centroids(vector<vector<Image<T>*>> clusters);
        Image<T>* find_newCentroid(vector<Image<T>*> cluster, int cluster_num);
        vector<Image<unsigned char>*> new_Centroids_Reduced(vector<vector<Image<T>*>> clusters, vector<Image<unsigned char>*> Data);
        Image<unsigned char>* find_newCentroid_Reduced(vector<Image<T>*> cluster, int cluster_num, vector<Image<unsigned char>*> Data);
        vector<vector<Image<T>*>> get_Clusters();
        void set_Clusters(vector<vector<Image<T>*>> clusters);
        vector<vector<Image<T>*>> make_Clusters();
        vector<double> compute_Silhouette();
        vector<double> compute_Silhouette_Reduced(vector<Image<unsigned char>*> Data, Distance_Function<unsigned char> original_distance_Function);
        long long int compute_Objective_Function();
        long long int compute_Objective_Function_Reduced(vector<Image<unsigned char>*> Data, Distance_Function<unsigned char> original_distance_Function);
};

#endif