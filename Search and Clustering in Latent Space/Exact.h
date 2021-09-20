#ifndef EXACT_H
#define EXACT_H

#include <vector>
#include "Technique.h"

template <typename T>
class Exact: public Technique<T>{
    private:
        /* For each query (first vector), a vector with the distances to each of the data */
        vector<vector<distancePair<T>*>> allDistances;
        /* Calculates the distances between each query and each datum once, so that the other functions 
        that find the nearest neighbors simply parse this structure of distances */
        void calculate_allDistances(Distance_Function<T> distance_Function);
        /* <------------ To find an appropriate value for w ------------> */
        //vector<vector<distancePair*>> allDataDistances;
        //void calculate_allDataDistances(Distance_Function distance_Function);
    public:
        Exact(int arg_k, int arg_N, double arg_r, vector<Image<T>*>& arg_data, vector<Image<T>*>& arg_queries);
        ~Exact();
        /* Returns a vector with the single nearest neighbor (from Data) of each query (from Queries)*/
        vector<distancePair<T>*> calculate_Nearest(Distance_Function<T> distance_Function);
        /* For each N (first vector), returns a vector of the N-th nearest neighbor for each query (second vector) */
        vector<vector<distancePair<T>*>> calculate_N_Nearest(Distance_Function<T> distance_Function);
        /* For each query (first vector), returns a vector of the data that lie in range "r" of that query */
        vector<vector<distancePair<T>*>> calculate_in_Range(Distance_Function<T> distance_Function);
        /* Multiplies "r" by 2 (not used in Exact) */
        void modify_radii();
        /* Modifies the queryset */
        void new_Queries(vector<Image<T>*>& newQueries);
        /* <------------ To find an appropriate value for w ------------> */
        //vector<distancePair*> calculate_Data_Nearest(Distance_Function distance_Function);
};

#endif