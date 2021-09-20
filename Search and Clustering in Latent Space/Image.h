#ifndef IMAGE_H
#define IMAGE_H

#include <vector>

using namespace std;

template <typename T>
class Image {
    private:
        vector<T> pixels;
        int length;
        int order; //the order of insertion into the vector of images (order read)
    public:
        Image(vector<T> arg_pixels, int arg_order);
        ~Image();
        vector<T> get_pixels();
        int get_order();
        int get_length();
};

#endif