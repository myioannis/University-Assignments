#include "Technique.h"

Technique::Technique(int arg_k, int arg_N, double arg_r, vector<Image*>& arg_data, vector<Image*>& arg_queries): k(arg_k), N(arg_N), r(arg_r), Data(arg_data), Queries(arg_queries)
{
    
}

Technique::~Technique()
{

}

bool Technique::compare_distancePair(distancePair* pair1, distancePair* pair2){
    return pair1->first < pair2->first;
}

int Technique::find_In_Neighbors(vector<distancePair*> best_candidates, Image* img)
{
    for (int i = 0; i < best_candidates.size(); i++)
        if (best_candidates[i]->second == img)
            return 1;
    return 0;
}

vector<chrono::duration<double>> Technique::get_Times()
{
    return Times;
}

double Technique::get_r()
{
    return r;
}