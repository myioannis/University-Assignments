#ifndef HYPERCUBE_H
#define HYPERCUBE_H

#include "Technique.h"
#include <random>
#include <unordered_map>

template <typename T>
class Hypercube: public Technique<T> {
    private:
        int w;
        int d;
        int M;
        int probes;
        vector<long int> m;
        long long int h_M;
    
        // vector of independent, real, uniformly distributed s values for each h function in each hash table
        vector<vector<int>> s;
        // vector of hash table that maps an h function to a 0 or 1 bit for each k
        vector<unordered_map<int, int>> f_functions;
        // hash table that maps binary strings to images. unordered_multimap allows multiple Images to one binary string
        // which allows collisions in a bucket.
        unordered_multimap<string, Image<T>*> hash_table;
        // size of hash table
        int ht_size;

        // vector of best distances paired with their corresponding nearest dataset candidates for each query
        vector<vector<distancePair<T>*>> best_nearest_candidates;
        // vector of best distances paired with their corresponding nearest dataset candidates in R range for each query
        vector<vector<distancePair<T>*>> best_range_candidates;

    public:
        Hypercube(int arg_k, int arg_M, int arg_N, int arg_d, int arg_probes, double arg_r, vector<Image<T>*>& arg_data, vector<Image<T>*>& arg_queries);
        ~Hypercube();

        int calculate_ai(int p_i, double s_i, int w);
        int calculate_hi(vector<T> p, int i);
        string calculate_f(vector<T> p);

        void generate_Hamming_String(vector<string>& vertices, string vertex, int len, int hamming_distance);
        vector<string> calculate_Neighbor_Vertices(string vertex);
        void insert_To_Hash_Table();

        vector<distancePair<T>*> calculate_Nearest(Distance_Function<T> dist_function);
        // for each query (first vector), returns a vector of its N nearest neighbors
        vector<vector<distancePair<T>*>> calculate_N_Nearest(Distance_Function<T> dist_function);
        // for each query (first vector), returns a vector of its nearest neighbors within range r
        vector<vector<distancePair<T>*>> calculate_in_Range(Distance_Function<T> dist_function);
        // multiplies the radii (r) by 2 (used in reverse assignment with range search in clustering)
        void modify_radii();
        // modifies the queryset
        void new_Queries(vector<Image<T>*>& newQueries);
};

#endif