#ifndef IMAGE_H
#define IMAGE_H

#include <vector>

using namespace std;

class Image {
    private:
        vector<unsigned char> pixels;
        int length;
        int order; //the order of insertion into the vector of images (order read)
    public:
        Image(vector<unsigned char> arg_pixels, int arg_order);
        ~Image();
        vector<unsigned char> get_pixels();
        int get_order();
        int get_length();
};

#endif