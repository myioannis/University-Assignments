#ifndef TECHNIQUE_H
#define TECHNIQUE_H

#include <vector>
#include <chrono>
#include "Image.h"
#include "Utils.h"

using namespace std;

typedef pair<double,Image*> distancePair;

class Technique {
    protected:
        //Members
        int k, N;
        double r;
        vector<Image*>& Data;
        vector<Image*>& Queries;

        vector<chrono::duration<double>> Times;
        //Methods
        static bool compare_distancePair(distancePair* pair1, distancePair* pair2);
    public:
        Technique(int arg_k, int arg_N, double arg_r, vector<Image*>& arg_data, vector<Image*>& arg_queries);
        virtual ~Technique();

        vector<chrono::duration<double>> get_Times();
        double get_r();
        int find_In_Neighbors(vector<distancePair*> best_candidates, Image* img);

        virtual vector<distancePair*> calculate_Nearest(Distance_Function dist_function) = 0;
        virtual vector<vector<distancePair*>> calculate_N_Nearest(Distance_Function dist_function) = 0;
        virtual vector<vector<distancePair*>> calculate_in_Range(Distance_Function dist_function) = 0;
        virtual void modify_radii() = 0;
        virtual void new_Queries(vector<Image*>& newQueries) = 0;
};

#endif