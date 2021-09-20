#include "Hypercube.h"
#include <limits>
#include <cmath>
#include <algorithm>

Hypercube::Hypercube(int arg_k, int arg_M, int arg_N, int arg_d, int arg_probes, double arg_r, vector<Image*>& arg_data, vector<Image*>& arg_queries): Technique(arg_k, arg_N, arg_r, arg_data, arg_queries), M(arg_M), d(arg_d), probes(arg_probes), w(10684)
{
    h_M = (long long int)pow(2, 32 / k);

    // generate k*d si, where i=0...d-1 numbers for each h in each table
    random_device rd;
    default_random_engine generator(rd());
    uniform_real_distribution<double> distribution(0.0, w);

    for (int i = 0; i < k; i++)
    {
        vector<int> temp_s; // s variables for current h function
        for (int j = 0; j < d; j++)
        {
            double s_j = distribution(generator);
            temp_s.push_back(s_j);
        }

        s.push_back(temp_s);
    }

    // calculate (m^i)%M beforehand, since m and M are constants, where i = 0...d-1
    // get results starting from last elements, so that 0-th element = (m^(d-1))%M and so on
    long long int my_m = (long long int)pow(2, 32) - 3;
    for (int i = d - 1; i >= 0; i--)
        m.push_back(modular_pow(my_m, i, h_M));

    // initialize hash table
    f_functions.resize(k);
    ht_size = k;
    insert_To_Hash_Table();
}

Hypercube::~Hypercube()
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


int Hypercube::calculate_ai(int p_i, double s_i, int w)
{
    return floor((p_i - s_i) / w);
}

int Hypercube::calculate_hi(vector<unsigned char> p, int h)
{
    int total_h = 0;
    int h_i = 0;
    for (int i = 0; i < d-1; i++)
    {
        int a_i = calculate_ai(p[i], s[h][i], w);

        int a_mod = mod(a_i, h_M);
        h_i = mod(a_mod * m[i], h_M);
        total_h += h_i;
    }
    return mod(total_h, h_M);
}

string Hypercube::calculate_f(vector<unsigned char> p)
{
    int h_i;
    string bit_string;

    random_device rd;
    default_random_engine generator(rd());
    uniform_int_distribution<int> distribution(0, 1);

    for (int i = 0; i < k; i++)
    {
        h_i = calculate_hi(p, i);
        if (!f_functions[i].count(h_i))//if h function has not been mapped to 0 or 1
        {
            f_functions[i].insert(make_pair(h_i, distribution(generator)));//map it to 0 or 1 and add it to the hash table
        }
        
        bit_string += to_string(f_functions[i].find(h_i)->second);//concatenate the 0 or 1 bit to the binary string
    }

    return bit_string;
}

// generates a hamming string based on a vertex string
void Hypercube::generate_Hamming_String(vector<string>& vertices, string vertex, int len, int hamming_distance)
{
    if (hamming_distance == 0)
    {
        vertices.push_back(vertex);
        return;
    }

    if (len < 0) return;
    // flip current bit
    vertex[len] = vertex[len] == '0' ? '1' : '0';
    generate_Hamming_String(vertices, vertex, len-1, hamming_distance-1);
    
    // or don't flip it (flip it again to undo)
    vertex[len] = vertex[len] == '0' ? '1' : '0';
    generate_Hamming_String(vertices, vertex, len-1, hamming_distance);
}

// generates neighbor vertices by generating all hamming strings
vector<string> Hypercube::calculate_Neighbor_Vertices(string vertex)
{
    vector<string> vertices;
    vertices.push_back(vertex);

    int len = vertex.length();
    int hamming_distance = len;

    string temp_vertex = vertex;

    for (int i = 1; i <= hamming_distance && vertices.size() < probes; ++i)
    {
        generate_Hamming_String(vertices, temp_vertex, len-1, i);
    }

    return vertices;
}

void Hypercube::insert_To_Hash_Table()
{
    vector<Image*> images = Data;
    int numOfImages = images.size();

    for (int img = 0; img < numOfImages; img++)
    {
        vector<unsigned char> pixels = images[img]->get_pixels();
        string binary_string = calculate_f(pixels); //calculate bucket number aka binary string
        hash_table.insert(make_pair(binary_string, images[img]));   //and add image to bucket indicated by the binary string
    }
}



vector<distancePair*> Hypercube::calculate_Nearest(Distance_Function dist_function)
{

}

vector<vector<distancePair*>> Hypercube::calculate_N_Nearest(Distance_Function dist_function)
{
    int k_neighbors = N;

    // initialize k best distances to max double
    for (int q = 0; q < Queries.size(); q++)
    {
        vector<distancePair*> temp_pairs;
        temp_pairs.reserve(k_neighbors);
        for (int i = 0; i < k_neighbors; i++)
        {
            distancePair* new_pair = new distancePair(numeric_limits<double>::max(), Queries[q]);
            temp_pairs.push_back(new_pair);
        }
        best_nearest_candidates.push_back(temp_pairs);
    }

    for (int q = 0; q < Queries.size(); q++)
    {
        auto start = chrono::system_clock::now();

        // project query point to corresponding hypercube vertex
        vector<unsigned char> pixels = Queries[q]->get_pixels();
        string vertex = calculate_f(pixels);
        vector<string> vertices = calculate_Neighbor_Vertices(vertex);

        for (int i = 0; i < probes; i++)    //visit neighbor vertices by given number of probes
        {
            int images_checked = 0;
            auto range = hash_table.equal_range(vertices[i]);

            for (auto p = range.first; p != range.second && images_checked < M; p++) // for each item p in bucket until M images have been checked
            {
                double dist = dist_function(Queries[q], p->second);
                if (dist < best_nearest_candidates[q][k_neighbors-1]->first)
                {
                    best_nearest_candidates[q][k_neighbors-1]->first = dist;
                    best_nearest_candidates[q][k_neighbors-1]->second = p->second;
                    
                    if (k_neighbors > 1)
                        sort(best_nearest_candidates[q].begin(), best_nearest_candidates[q].end(), compare_distancePair);
                }

                images_checked++;
            }
        }

        auto end = chrono::system_clock::now();
        auto duration = end-start;
        Times.push_back(duration);
    }

    return best_nearest_candidates;
}

vector<vector<distancePair*>> Hypercube::calculate_in_Range(Distance_Function dist_function)
{
    best_range_candidates.resize(Queries.size());

    for (int q = 0; q < Queries.size(); q++)
    {
        vector<unsigned char> pixels = Queries[q]->get_pixels();
        string vertex = calculate_f(pixels);
        vector<string> vertices = calculate_Neighbor_Vertices(vertex);

        for (int i = 0; i < probes; i++)
        {
            int images_checked = 0;
            auto range = hash_table.equal_range(vertices[i]);

            for (auto p = range.first; p != range.second && images_checked < M; p++)
            {
                double dist = dist_function(Queries[q], p->second);
                if (dist < r)
                {
                    distancePair* b = new distancePair(dist, p->second);
                    best_range_candidates[q].push_back(b);
                }

                images_checked++;
            }
        }

        sort(best_range_candidates[q].begin(), best_range_candidates[q].end(), compare_distancePair);
    }

    return best_range_candidates;
}

void Hypercube::modify_radii(){
    r = r*2;
}

void Hypercube::new_Queries(vector<Image*>& newQueries){
    Queries = newQueries;
}