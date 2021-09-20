#include "LSH.h"
#include "Technique.h"
#include <limits>
#include <algorithm>

template <typename T>
LSH<T>::LSH(int arg_k, int arg_L, int arg_N, int arg_d, double arg_r, vector<Image<T>*>& arg_data, vector<Image<T>*>& arg_queries): Technique<T>(arg_k, arg_N, arg_r, arg_data, arg_queries), L(arg_L), d(arg_d), w(10684)
{
    M = (long long int)pow(2, 32 / this->k);

    // generate L*k*d independent, real, uniformly distributed si, where i=0...d-1 numbers for each h function in each hash table
    random_device rd;
    default_random_engine generator(rd());
    uniform_real_distribution<double> distribution(0.0, w);

    for (int table = 0; table < L; table++)
    {
        vector<vector<int>> temp_table_h;   // h functions (aka vectors of s variables) for current table
        for (int i = 0; i < this->k; i++)
        {
            vector<int> temp_s; // s variables for current h function
            for (int j = 0; j < d; j++)
            {
                double s_j = distribution(generator);
                temp_s.push_back(s_j);
            }

            temp_table_h.push_back(temp_s);
        }
        s.push_back(temp_table_h);
    }

    // calculate (m^i)%M beforehand, since m and M are constants, where i = 0...d-1
    // get results starting from last elements, so that 0-th element = (m^(d-1))%M and so on
    long long int my_m = (long long int)pow(2, 32) - 3;
    for (int i = d - 1; i >= 0; i--)
        m.push_back(modular_pow(my_m, i, M));

    // initialize L hash tables
    hash_tables.resize(L);
    ht_size = this->Data.size() / 8;
    insert_To_Hash_Tables();
}

template <typename T>
LSH<T>::~LSH()
{
    for (int query_num = 0; query_num < best_nearest_candidates.size(); query_num++){
        for (int neighbor_num = 0; neighbor_num < best_nearest_candidates[query_num].size(); neighbor_num++){
            delete best_nearest_candidates[query_num][neighbor_num];
        }
    }

    for (int query_num = 0; query_num < best_range_candidates.size(); query_num++){
        for (int neighbor_num = 0; neighbor_num < best_range_candidates[query_num].size(); neighbor_num++){
            delete best_range_candidates[query_num][neighbor_num];
        }
    }
}

template <typename T>
int LSH<T>::calculate_ai(int p_i, double s_i, int w)
{
    return floor((p_i - s_i) / w);
}

template <typename T>
int LSH<T>::calculate_hi(vector<T> p, int table, int h)
{
    int total_h = 0;
    int h_i = 0;
    for (int i = 0; i < d-1; i++)
    {
        int a_i = calculate_ai(p[i], s[table][h][i], w);

        int a_mod = mod(a_i, M);
        h_i = mod(a_mod * m[i], M);
        total_h += h_i;
    }
    return mod(total_h, M);
}

template <typename T>
unsigned int LSH<T>::calculate_g(vector<T> p, int table)
{
    unsigned int g_ip = 0;
    unsigned int h_i;
    // g function is the product of concatenating all h functions
    for (int i = this->k-1; i >= 0; i--)
    {
        h_i = calculate_hi(p, table, i);
        g_ip |= h_i;
        if (i != 0)
            g_ip <<= 8;
    }
    return g_ip;
}

template <typename T>
void LSH<T>::insert_To_Hash_Tables()
{
    vector<Image<T>*> images = this->Data;
    int numOfImages = images.size();

    for (int img = 0; img < numOfImages; img++)
    {
        for (int table = 0; table < L; table++)
        {
            vector<T> pixels = images[img]->get_pixels();
            unsigned int g = calculate_g(pixels, table) % ht_size;  //calculate bucket number g, which is g function mod hash table size
            hash_tables[table].insert(make_pair(g, images[img]));   //and add image to bucket indicated by g
        }
    }
}

template <typename T>
vector<distancePair<T>*> LSH<T>::calculate_Nearest(Distance_Function<T> dist_function)
{

}

template <typename T>
vector<vector<distancePair<T>*>> LSH<T>::calculate_N_Nearest(Distance_Function<T> dist_function)
{
    int k_neighbors = this->N;
    
    // initialize k best distances to max double
    for (int q = 0; q < this->Queries.size(); q++)
    {
        vector<distancePair<T>*> temp_pairs;
        temp_pairs.reserve(k_neighbors);
        for (int i = 0; i < k_neighbors; i++)
        {
            distancePair<T>* new_pair = new distancePair<T>(numeric_limits<double>::max(), this->Queries[q]);
            temp_pairs.push_back(new_pair);
        }
        best_nearest_candidates.push_back(temp_pairs);
    }

    for (int q = 0; q < this->Queries.size(); q++)
    {
        auto start = chrono::system_clock::now();
        for (int table = 0; table < L; table++)     // for i from 1 to L do
        {
            vector<T> pixels = this->Queries[q]->get_pixels();
            unsigned int g = calculate_g(pixels, table) % ht_size;
            auto range = hash_tables[table].equal_range(g);
            
            for (auto p = range.first; p != range.second; p++)  // for each item p in bucket gi(q) do
            {
                double dist = dist_function(this->Queries[q], p->second);  // dist(q,p)
                if (dist < best_nearest_candidates[q][k_neighbors-1]->first && !this->find_In_Neighbors(best_nearest_candidates[q], p->second))    // if dist(q,p) < db, where db k-th best distance
                {
                    best_nearest_candidates[q][k_neighbors-1]->first = dist;
                    best_nearest_candidates[q][k_neighbors-1]->second = p->second;
                    
                    if (k_neighbors > 1)
                        sort(best_nearest_candidates[q].begin(), best_nearest_candidates[q].end(), this->compare_distancePair);
                }

                //if (hash_tables[table].count(g) > 10* L)    // if large number of retrieved items
                //    break;
            }
        }
        auto end = chrono::system_clock::now();
        auto duration = end-start;
        this->Times.push_back(duration);
    }

    return best_nearest_candidates;
}

template <typename T>
vector<vector<distancePair<T>*>> LSH<T>::calculate_in_Range(Distance_Function<T> dist_function)
{
    best_range_candidates.resize(this->Queries.size());

    for (int q = 0; q < this->Queries.size(); q++)
    {
        for (int table = 0; table < L; table++)     // for i from 1 to L do
        {
            vector<T> pixels = this->Queries[q]->get_pixels();
            unsigned int g = calculate_g(pixels, table) % ht_size;
            auto range = hash_tables[table].equal_range(g);
            
            for (auto p = range.first; p != range.second; p++)  // for each item p in bucket gi(q) do
            {
                double dist = dist_function(this->Queries[q], p->second);  // dist(q,p)
                if (dist < this->r && !this->find_In_Neighbors(best_range_candidates[q], p->second))    // if dist(q,p) <= radius
                {
                    distancePair<T>* b = new distancePair<T>(dist, p->second);
                    best_range_candidates[q].push_back(b);
                }

                //if (hash_tables[table].count(g) > 20 * L)    // if large number of retrieved items
                //    break;
            }
        }

        sort(best_range_candidates[q].begin(), best_range_candidates[q].end(), this->compare_distancePair);
    }

    return best_range_candidates;
}

template <typename T>
void LSH<T>::modify_radii(){
    this->r = this->r*2;
}

template <typename T>
void LSH<T>::new_Queries(vector<Image<T>*>& newQueries){
    this->Queries = newQueries;
}

template class LSH<unsigned char>;
template class LSH<unsigned short>;