#include "Technique.h"

template <typename T>
Technique<T>::Technique(int arg_k, int arg_N, double arg_r, vector<Image<T>*>& arg_data, vector<Image<T>*>& arg_queries): k(arg_k), N(arg_N), r(arg_r), Data(arg_data), Queries(arg_queries)
{
    
}

template <typename T>
Technique<T>::~Technique()
{

}

template <typename T>
bool Technique<T>::compare_distancePair(distancePair<T>* pair1, distancePair<T>* pair2){
    return pair1->first < pair2->first;
}

template <typename T>
int Technique<T>::find_In_Neighbors(vector<distancePair<T>*> best_candidates, Image<T>* img)
{
    for (int i = 0; i < best_candidates.size(); i++)
        if (best_candidates[i]->second == img)
            return 1;
    return 0;
}

template <typename T>
vector<chrono::duration<double>> Technique<T>::get_Times()
{
    return Times;
}

template <typename T>
double Technique<T>::get_r()
{
    return r;
}

template class Technique<unsigned char>;
template class Technique<unsigned short>;