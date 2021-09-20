#ifndef TECHNIQUE_H
#define TECHNIQUE_H

#include <vector>
#include <chrono>
#include "Image.h"
#include "Utils.h"

using namespace std;

template <typename T>
using distancePair = pair<double,Image<T>*>;

template <typename T>
class Technique {
    protected:
        //Members
        int k, N;
        double r;
        vector<Image<T>*>& Data;
        vector<Image<T>*>& Queries;

        vector<chrono::duration<double>> Times;
        //Methods
        static bool compare_distancePair(distancePair<T>* pair1, distancePair<T>* pair2);
    public:
        Technique(int arg_k, int arg_N, double arg_r, vector<Image<T>*>& arg_data, vector<Image<T>*>& arg_queries);
        virtual ~Technique();

        vector<chrono::duration<double>> get_Times();
        double get_r();
        int find_In_Neighbors(vector<distancePair<T>*> best_candidates, Image<T>* img);

        virtual vector<distancePair<T>*> calculate_Nearest(Distance_Function<T> dist_function) = 0;
        virtual vector<vector<distancePair<T>*>> calculate_N_Nearest(Distance_Function<T> dist_function) = 0;
        virtual vector<vector<distancePair<T>*>> calculate_in_Range(Distance_Function<T> dist_function) = 0;
        virtual void modify_radii() = 0;
        virtual void new_Queries(vector<Image<T>*>& newQueries) = 0;
};

#endif