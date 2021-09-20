#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <limits>
#include "Cluster.h"
#include "Technique.h"
#include "Exact.h"
#include "LSH.h"
#include "Hypercube.h"
#include "Image.h"

using namespace std;

template <typename T>
Cluster<T>::Cluster(int arg_K, vector<Image<T>*>& arg_nC, Distance_Function<T> arg_distance_Function): K(arg_K), nonCentroids(arg_nC), distance_Function(arg_distance_Function) {
    Centroids.reserve(K);
}

template <typename T>
Cluster<T>::Cluster(int arg_K, int arg_w, int arg_L, int arg_k, int arg_d, int arg_M, int arg_probes, int arg_r, string arg_method, vector<Image<T>*>& arg_nC, Distance_Function<T> arg_distance_Function): K(arg_K),w(arg_w),L(arg_L),k(arg_k),d(arg_d),M(arg_M),probes(arg_probes),r(arg_r),method(arg_method),nonCentroids(arg_nC),distance_Function(arg_distance_Function){
    // <--Initialization++-->
    //Choose a (first) centroid uniformly at random 
    random_device rd;
    default_random_engine generator(rd());
    uniform_int_distribution<int> distribution(0,nonCentroids.size()-1);
    int randomIndex = distribution(generator);
    Image<T>* firstCentroid = new Image<T>(nonCentroids[randomIndex]->get_pixels(),0);
    Centroids.push_back(firstCentroid);
    newCentroids.push_back(firstCentroid);
    //Until we choose all K initial centroids
    vector<double> D; D.reserve(nonCentroids.size());
    vector<double> P; P.reserve(nonCentroids.size()+1);
    vector<distancePair<T>*> minDistances; minDistances.reserve(nonCentroids.size());
    for (int K_num=1; K_num<K; K_num++){
        //For each nonCentroid i compute D(i)=min dist to some centroid
        Exact<T> myExact(k,1,r,Centroids,nonCentroids); //In Exact -> Data=Centroids, Queries=nonCentroids
        minDistances = myExact.calculate_Nearest(distance_Function);
        //Assign the minimum distances to vector D
        D.clear();
        for(int i=0; i<minDistances.size(); i++) D.push_back(minDistances[i]->first); 
        //Calculate the vector P
        P.clear();
        P.push_back(0.0); //P[0]=0
        for(int i=0; i<D.size(); i++) P.push_back(P[i]+pow(D[i],2));
        //Pick a uniformly distributed float x in [0, P(n − t)] 
        uniform_real_distribution<double> distribution(0,P.back());
        double randomFloat = distribution(generator);
        //Find the corresponding index for the new Centroid
        int newCentroidIndex;
        for (int i=1; i<P.size(); i++){
            if (randomFloat>P[i-1] && randomFloat<=P[i]) newCentroidIndex = i;
        }
        //New centroid
        Image<T>* newCentroid = new Image<T>(nonCentroids[newCentroidIndex]->get_pixels(),K_num);
        Centroids.push_back(newCentroid);
        newCentroids.push_back(newCentroid);
    }
    cout << "Centroids: " << Centroids.size() << endl;
    for (int i=0; i<Centroids.size(); i++){
        for (int j=Centroids.size()-1; j>i; j--){
            cout << "D[C"<<i<<"->C"<<j<<"]="<< distance_Function(Centroids[i],Centroids[j]) << endl;;
        }
    }
}

template <typename T>
Cluster<T>::~Cluster(){
    for(int image_num=0; image_num<newCentroids.size(); image_num++){
        delete newCentroids[image_num];
    }
}

template <typename T>
T Cluster<T>::median_ofDimension(vector<T>& allValuesOfDimension){
    sort(allValuesOfDimension.begin(),allValuesOfDimension.end());
    int dimensions = allValuesOfDimension.size();
    if (dimensions % 2 != 0) //if the size of the array is odd
        return allValuesOfDimension[dimensions / 2]; 
    //else if the size of the array is even (we return the average of the n/2 element and the n/2-1 element)
    return ceil((double)(allValuesOfDimension[(dimensions-1)/2] + allValuesOfDimension[dimensions/2])/ 2.0); 
}

template <typename T>
Image<T>* Cluster<T>::find_newCentroid(vector<Image<T>*> cluster, int cluster_num){
    vector<T> MedianPixels; //A vector of pixels
    MedianPixels.reserve(cluster[0]->get_length()); //size of an Image's pixels
    for (int dimension=0; dimension<cluster[0]->get_length(); dimension++){
        vector<T> allValuesOfDimension;
        allValuesOfDimension.reserve(cluster.size()); //size of the number of images in the cluster
        for (int image_num=0; image_num<cluster.size(); image_num++){ 
            allValuesOfDimension.push_back(cluster[image_num]->get_pixels()[dimension]);
        }
        MedianPixels.push_back(median_ofDimension(allValuesOfDimension));
    }
    Image<T>* newCentroid = new Image<T>(MedianPixels,cluster_num);
    return newCentroid;
}

//It computes the <Median> for each dimension of each nonCentroid of a cluster
template <typename T>
void Cluster<T>::new_Centroids(vector<vector<Image<T>*>> clusters){
    //compute medians and assign them to Centroids
    vector<Image<T>*> medianCentroids; medianCentroids.reserve(Centroids.size());
    for (int cluster_num=0; cluster_num<clusters.size(); cluster_num++){
        medianCentroids.push_back(find_newCentroid(clusters[cluster_num],cluster_num));
    }
    Centroids.clear();
    for (int cluster_num=0; cluster_num<clusters.size(); cluster_num++){
        Centroids.push_back(medianCentroids[cluster_num]);
        newCentroids.push_back(medianCentroids[cluster_num]);
    }
}

template <typename T>
unsigned char Cluster<T>::median_ofDimension_Reduced(vector<unsigned char>& allValuesOfDimension){
    sort(allValuesOfDimension.begin(),allValuesOfDimension.end());
    int dimensions = allValuesOfDimension.size();
    if (dimensions % 2 != 0) //if the size of the array is odd
        return allValuesOfDimension[dimensions / 2]; 
    //else if the size of the array is even (we return the average of the n/2 element and the n/2-1 element)
    return ceil((double)(allValuesOfDimension[(dimensions-1)/2] + allValuesOfDimension[dimensions/2])/ 2.0); 
}

template <typename T>
Image<unsigned char>* Cluster<T>::find_newCentroid_Reduced(vector<Image<T>*> cluster, int cluster_num, vector<Image<unsigned char>*> Data){
    vector<unsigned char> MedianPixels; //A vector of pixels
    MedianPixels.reserve(Data[0]->get_length()); //size of an Image's pixels in original space
    for (int dimension=0; dimension<Data[0]->get_length(); dimension++){
        vector<unsigned char> allValuesOfDimension;
        allValuesOfDimension.reserve(cluster.size()); //size of the number of images in the cluster
        for (int image_num=1; image_num<cluster.size(); image_num++){
            allValuesOfDimension.push_back(Data[cluster[image_num]->get_order()]->get_pixels()[dimension]);
        }
        MedianPixels.push_back(median_ofDimension_Reduced(allValuesOfDimension));
    }
    Image<unsigned char>* newCentroid = new Image<unsigned char>(MedianPixels,cluster_num);
    return newCentroid;
}

//It computes the <Median> for each dimension of each nonCentroid of a cluster
template <typename T>
vector<Image<unsigned char>*> Cluster<T>::new_Centroids_Reduced(vector<vector<Image<T>*>> clusters, vector<Image<unsigned char>*> Data){
    //compute medians and assign them to Centroids
    vector<Image<unsigned char>*> medianCentroids; medianCentroids.reserve(Centroids.size());
    for (int cluster_num=0; cluster_num<clusters.size(); cluster_num++){
        medianCentroids.push_back(find_newCentroid_Reduced(clusters[cluster_num], cluster_num, Data));
    }
    return medianCentroids;
}

template <typename T>
vector<vector<Image<T>*>> Cluster<T>::assignment(vector<distancePair<T>*>& closestCentroids){
    vector<vector<Image<T>*>> clusters; clusters.reserve(Centroids.size());
    for (int centroid_num = 0; centroid_num<Centroids.size(); centroid_num++){
        vector<Image<T>*> pointsOfCluster; //nonCentroids assigned to a centroid
        pointsOfCluster.push_back(Centroids[centroid_num]); //the first item is the centroid itself
        for (int nonCentroid_num = 0; nonCentroid_num<closestCentroids.size(); nonCentroid_num++){
            if (closestCentroids[nonCentroid_num]->second->get_order()==Centroids[centroid_num]->get_order()){
                pointsOfCluster.push_back(nonCentroids[nonCentroid_num]);
            }
        }
        clusters.push_back(pointsOfCluster);
    }
    return clusters;
}

template <typename T>
vector<vector<Image<T>*>> Cluster<T>::Approximate_assignment(vector<vector<distancePair<T>*>>& closestNeighbors, unordered_map<Image<T>*, pair<int, double>>& assigned, int& numOfChanges){
    vector<vector<Image<T>*>> clusters;
    clusters.resize(Centroids.size());

    for (int centroid_num = 0; centroid_num<Centroids.size(); centroid_num++){//for each centroid
        clusters[centroid_num].push_back(Centroids[centroid_num]);

        for (int neighbor_num = 0; neighbor_num<closestNeighbors[centroid_num].size(); neighbor_num++){//for each centroid's neighbor
            double distance = closestNeighbors[centroid_num][neighbor_num]->first;
            Image<T>* nonCentroid = closestNeighbors[centroid_num][neighbor_num]->second;

            typename unordered_map<Image<T>*, pair<int, double>>::const_iterator it = assigned.find(nonCentroid);
            if (it == assigned.end()){//if image is not already assigned to a cluster
                assigned.insert({nonCentroid, make_pair(centroid_num, distance)});  //add it to hash table with its corresponding cluster and distance from its centroid
            }
            else {  //if image is already assigned to a cluster
                int oldCentroid = it->second.first;
                double oldDistance = it->second.second;
                if (distance < oldDistance){    //assign to this cluster if this centroid is closer to the image than the other centroid
                    assigned.erase(it);
                    assigned.insert({nonCentroid, make_pair(centroid_num, distance)});
                    numOfChanges++;
                }
            }
        }
    }
    return clusters;
}

//checks whether only a small portion of the data (nonCentroids) changed clusters
template <typename T>
bool Cluster<T>::converged(vector<distancePair<T>*> closestCentroids,vector<distancePair<T>*> prevClosestCentroids){
    double threshold = nonCentroids.size() * 0.02;
    int numOfChanges = 0;
    if (!prevClosestCentroids.empty()){ //if this is NOT the first iteration
        for (int nonCentroid_num=0; nonCentroid_num<nonCentroids.size(); nonCentroid_num++){
            if (closestCentroids[nonCentroid_num]->second->get_order()!=prevClosestCentroids[nonCentroid_num]->second->get_order()){
                numOfChanges++;
            }
        }
        cout << "numOfChanges=" << numOfChanges << " Threshold=" << threshold << endl;
        if (numOfChanges<threshold) return true;
    }
    return false;
}

//checks whether only a small portion of the data (nonCentroids) changed clusters
template <typename T>
bool Cluster<T>::Approximate_converged(unordered_map<Image<T>*, pair<int, double>>& assigned, int& oldUnassigned, int iteration, int numOfChanges){
    int numOfUnassigned = 0;
    if (iteration > 1){ //if this is NOT the first iteration
        for (int nonCentroid_num=0; nonCentroid_num<nonCentroids.size(); nonCentroid_num++){//for all non centroids
            if (assigned.find(nonCentroids[nonCentroid_num]) == assigned.end())//non centroid is not assigned to a cluster
                numOfUnassigned++;
        }
        cout << "numOfChanges=" << numOfChanges << " numOfUnassigned=" << numOfUnassigned << " oldUnassigned=" << oldUnassigned << " assigned=" << assigned.size() << endl;
        if (iteration == 20 || numOfUnassigned >= oldUnassigned){//stop if there are no new assignments
            return true;
        }
        oldUnassigned = numOfUnassigned;
    }
    return false;
}

template <typename T>
double Cluster<T>::initialize_Radius(){
    //find initial radius (min distance between centers / 2)
    double radius = numeric_limits<double>::max();
    for (int i = 0; i < Centroids.size(); i++){
        for (int j = 0; j < Centroids.size(); j++){
            if (i == j) continue;
            else {
                double dist = distance_Function(Centroids[i], Centroids[j]);
                if (dist < radius) radius = dist;
            }
        }
    }
    return radius / 2;
}

template <typename T>
vector<vector<Image<T>*>> Cluster<T>::get_Clusters(){
    return Clusters;
}

template <typename T>
void Cluster<T>::set_Clusters(vector<vector<Image<T>*>> clusters){
    Clusters = clusters;
    for (int cluster_num=0; cluster_num<Clusters.size(); cluster_num++){
        Clusters[cluster_num].insert(Clusters[cluster_num].begin(), Centroids[cluster_num]);
    }
}

template <typename T>
vector<vector<Image<T>*>> Cluster<T>::make_Clusters(){
    //Exact (Lloyd's) approach
    if (method=="Classic"){
        //closestCentroids: for each nonCentroid, its closest centroid
        vector<distancePair<T>*> closestCentroids; closestCentroids.reserve(nonCentroids.size());
        vector<distancePair<T>*> prevClosestCentroids; prevClosestCentroids.reserve(nonCentroids.size());
        Exact<T>* newExact = NULL;
        Exact<T>* prevExact = NULL;
        int iteration = 0;
        bool conv = false;
        do{
            cout << "Iteration: " << iteration << endl;
            if (iteration>0) new_Centroids(Clusters);
            prevExact = newExact;
            newExact = new Exact<T>(k,1,r,Centroids,nonCentroids); //In Exact -> Data=Centroids, Queries=nonCentroids
            prevClosestCentroids = closestCentroids;
            closestCentroids = newExact->calculate_Nearest(distance_Function);
            Clusters = assignment(closestCentroids);
            conv = converged(closestCentroids,prevClosestCentroids);
            iteration++;
            delete prevExact;
        }while(!conv);
        delete newExact;
    }
    //Approximate + Reverse approach
    else if (method=="LSH"){
        LSH<T> clusterLSH(k, L, 1, nonCentroids[0]->get_length(), initialize_Radius(), nonCentroids, Centroids);
        vector<vector<distancePair<T>*>> closestNeighbors;    //for each centroid, vector of neighbors within range r
        //hash table of assigned images to clusters, unordered map doesn't allow collisions thus each image is assigned to only 1 cluster
        unordered_map<Image<T>*, pair<int, double>> assigned;

        int iteration = 0;
        int numOfChanges = 0;
        int numOfUnassigned = nonCentroids.size();
        do{
            cout << "Iteration: " << iteration << endl;
            if (iteration>0){
                new_Centroids(Clusters);
                clusterLSH.new_Queries(Centroids);
            }

            //new assignement
            assigned.clear();

            closestNeighbors = clusterLSH.calculate_in_Range(distance_Function);
            numOfChanges = 0;
            Clusters = Approximate_assignment(closestNeighbors, assigned, numOfChanges);

            clusterLSH.modify_radii();
            iteration++;
        }while(!Approximate_converged(assigned, numOfUnassigned, iteration, numOfChanges));

        //algorithm has converged
        for (int nonCentroid_num=0; nonCentroid_num<nonCentroids.size(); nonCentroid_num++){
            typename unordered_map<Image<T>*, pair<int, double>>::const_iterator it = assigned.find(nonCentroids[nonCentroid_num]);
            if (it == assigned.end()){//if non centroid has not been assigned, assign it to the closest centroid
                double dist = numeric_limits<double>::max();
                int c = -1;
                for (int centroid_num = 0; centroid_num < Centroids.size(); centroid_num++){
                    double temp = distance_Function(nonCentroids[nonCentroid_num], Centroids[centroid_num]);
                    if (temp < dist){
                        dist = temp;
                        c = centroid_num;
                    }
                }
                Clusters[c].push_back(nonCentroids[nonCentroid_num]);
            }
            else {//if non centroid has been assigned, add it to its cluster
                int centroid_num = it->second.first;
                Clusters[centroid_num].push_back(it->first);
            }
        }
    }
    else if (method=="Hypercube"){
        Hypercube<T> clusterHypercube(k, M, 1, nonCentroids[0]->get_length(), probes, initialize_Radius(), nonCentroids, Centroids);
        vector<vector<distancePair<T>*>> closestNeighbors;
        unordered_map<Image<T>*, pair<int, double>> assigned;

        int iteration = 0;
        int numOfChanges = 0;
        int numOfUnassigned = nonCentroids.size();
        do{
            cout << "Iteration: " << iteration << endl;
            if (iteration>0){
                new_Centroids(Clusters);
                clusterHypercube.new_Queries(Centroids);
            }

            assigned.clear();

            closestNeighbors = clusterHypercube.calculate_in_Range(distance_Function);
            numOfChanges = 0;
            Clusters = Approximate_assignment(closestNeighbors, assigned, numOfChanges);

            clusterHypercube.modify_radii();
            iteration++;
        }while(!Approximate_converged(assigned, numOfUnassigned, iteration, numOfChanges));

        for (int nonCentroid_num=0; nonCentroid_num<nonCentroids.size(); nonCentroid_num++){
            typename unordered_map<Image<T>*, pair<int, double>>::const_iterator it = assigned.find(nonCentroids[nonCentroid_num]);
            if (it == assigned.end()){
                double dist = numeric_limits<double>::max();
                int c = -1;
                for (int centroid_num = 0; centroid_num < Centroids.size(); centroid_num++){
                    double temp = distance_Function(nonCentroids[nonCentroid_num], Centroids[centroid_num]);
                    if (temp < dist){
                        dist = temp;
                        c = centroid_num;
                    }
                }
                Clusters[c].push_back(nonCentroids[nonCentroid_num]);
            }
            else {
                int centroid_num = it->second.first;
                Clusters[centroid_num].push_back(it->first);
            }
        }
    }
    return Clusters;
}

template <typename T>
vector<double> Cluster<T>::compute_Silhouette(){
    //For each point "i" find its 2 nearest centroids ("i" is already assigned to the cluster of its first closest centroid)
    vector<vector<distancePair<T>*>> NNearestCentroids; //for each N, a vector of each Query's N-th neighbor
    Exact<T> myExact(k,2,r,Centroids,nonCentroids); //In Exact -> Data=Centroids, Queries=nonCentroids
    NNearestCentroids = myExact.calculate_N_Nearest(distance_Function);
    //For each datapoint "i" (nonCentroid)
    vector<double> a; a.reserve(nonCentroids.size());
    vector<double> b; b.reserve(nonCentroids.size());
    vector<double> s; s.reserve(nonCentroids.size());
    for (int i=0; i<nonCentroids.size(); i++){
        //Calculate a[i] (average distance of i to objects in same cluster)
        int cluster_num = NNearestCentroids[0][i]->second->get_order(); //the 0-th nearest centroid is its cluster
        double distanceSum = 0;
        for (int point_num=0; point_num<Clusters[cluster_num].size(); point_num++){
            distanceSum+=distance_Function(nonCentroids[i],Clusters[cluster_num][point_num]);
        }
        a.push_back(distanceSum/Clusters[cluster_num].size());
        //Calculate b[i] (average distance of i to objects in next best (neighbor) cluster)
        int next_cluster = NNearestCentroids[1][i]->second->get_order();
        distanceSum = 0;
        for (int point_num=0; point_num<Clusters[next_cluster].size(); point_num++){
            distanceSum+=distance_Function(nonCentroids[i],Clusters[next_cluster][point_num]);
        }
        b.push_back(distanceSum/Clusters[next_cluster].size());        
        //Calculate s[i]
        s.push_back( (b[i]-a[i])/max(a[i],b[i]) );
    }
    return s;
}

template <typename T>
vector<double> Cluster<T>::compute_Silhouette_Reduced(vector<Image<unsigned char>*> Data, Distance_Function<unsigned char> original_distance_Function){
    //For each point "i" find its 2 nearest centroids ("i" is already assigned to the cluster of its first closest centroid)
    vector<vector<distancePair<T>*>> NNearestCentroids; //for each N, a vector of each Query's N-th neighbor
    Exact<T> myExact(k,2,r,Centroids,nonCentroids); //In Exact -> Data=Centroids, Queries=nonCentroids
    NNearestCentroids = myExact.calculate_N_Nearest(distance_Function);
    //For each datapoint "i" (nonCentroid)
    vector<double> a; a.reserve(nonCentroids.size());
    vector<double> b; b.reserve(nonCentroids.size());
    vector<double> s; s.reserve(nonCentroids.size());
    for (int i=0; i<nonCentroids.size(); i++){
        //Calculate a[i] (average distance of i to objects in same cluster)
        int cluster_num = NNearestCentroids[0][i]->second->get_order(); //the 0-th nearest centroid is its cluster
        double distanceSum = 0;
        for (int point_num=1; point_num<Clusters[cluster_num].size(); point_num++){
            distanceSum+=original_distance_Function(Data[nonCentroids[i]->get_order()],Data[Clusters[cluster_num][point_num]->get_order()]);
        }
        a.push_back(distanceSum/Clusters[cluster_num].size());
        //Calculate b[i] (average distance of i to objects in next best (neighbor) cluster)
        int next_cluster = NNearestCentroids[1][i]->second->get_order();
        distanceSum = 0;
        for (int point_num=1; point_num<Clusters[next_cluster].size(); point_num++){
            distanceSum+=original_distance_Function(Data[nonCentroids[i]->get_order()],Data[Clusters[next_cluster][point_num]->get_order()]);
        }
        b.push_back(distanceSum/Clusters[next_cluster].size());        
        //Calculate s[i]
        s.push_back( (b[i]-a[i])/max(a[i],b[i]) );
    }
    return s;
}

template <typename T>
long long int Cluster<T>::compute_Objective_Function(){
    long long int obj_function = numeric_limits<long long int>::max();
    for (int cluster_num=0; cluster_num<Clusters.size(); cluster_num++){
        // For every cluster, calculate the total sum of distances between its centroid and its cluster points
        long long int cluster_sum = 0.0;
        for (int point_num=0; point_num<Clusters[cluster_num].size(); point_num++){
            cluster_sum += distance_Function(Centroids[cluster_num], Clusters[cluster_num][point_num]);
        }
        // Find the minimum distance sum across all clusters
        if (cluster_sum < obj_function) obj_function = cluster_sum;
    }
    return obj_function;
}

template <typename T>
long long int Cluster<T>::compute_Objective_Function_Reduced(vector<Image<unsigned char>*> Data, Distance_Function<unsigned char> original_distance_Function){
    long long int obj_function = numeric_limits<long long int>::max();
    // Find new centroids in original space
    vector<Image<unsigned char>*> medianCentroids = new_Centroids_Reduced(Clusters, Data);
    for (int cluster_num=0; cluster_num<Clusters.size(); cluster_num++){
        // For every cluster, calculate the total sum of distances between its centroid and its cluster points
        long long int cluster_sum = 0.0;
        for (int point_num=1; point_num<Clusters[cluster_num].size(); point_num++){
            cluster_sum += original_distance_Function(medianCentroids[cluster_num], Data[Clusters[cluster_num][point_num]->get_order()]);
        }
        // Find the minimum distance sum across all clusters
        if (cluster_sum < obj_function) obj_function = cluster_sum;
    }
    return obj_function;
}

template class Cluster<unsigned char>;
template class Cluster<unsigned short>;