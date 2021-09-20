#ifndef LSH_H
#define LSH_H

#include <random>
#include <unordered_map>
#include "Technique.h"
#include "Utils.h"

class LSH: public Technique {
    private:
        //LSH variables
        int w, d, L;
        vector<long int> m;
        long long int M;

        // vector of independent, real, uniformly distributed s values for each h function in each hash table
        vector<vector<vector<int>>> s;
        // vector of hash tables that map g functions to images. unordered_multimap allows multiple Images to one g function
        // which allows collisions in a bucket.
        vector<unordered_multimap<unsigned int, Image*>> hash_tables;
        // size of each hash table
        int ht_size;

        // vector of best distances paired with their corresponding nearest dataset candidates for each query
        vector<vector<distancePair*>> best_nearest_candidates;
        // vector of best distances paired with their corresponding nearest dataset candidates in R range for each query
        vector<vector<distancePair*>> best_range_candidates;

    public:
        LSH(int arg_k, int arg_L, int arg_N, int arg_d, double arg_r, vector<Image*>& arg_data, vector<Image*>& arg_queries);
        ~LSH();

        int calculate_ai(int p_i, double s_i, int w);
        int calculate_hi(vector<unsigned char> p, int table, int i);
        unsigned int calculate_g(vector<unsigned char> p, int table);

        void insert_To_Hash_Tables();

        vector<distancePair*> calculate_Nearest(Distance_Function dist_function);
        // for each query (first vector), returns a vector of its N nearest neighbors
        vector<vector<distancePair*>> calculate_N_Nearest(Distance_Function dist_function);
        // for each query (first vector), returns a vector of its nearest neighbors within range r
        vector<vector<distancePair*>> calculate_in_Range(Distance_Function dist_function);
        // multiplies the radii (r) by 2 (used in reverse assignment with range search in clustering)
        void modify_radii();
        // modifies the queryset
        void new_Queries(vector<Image*>& newQueries);
};

#endif