#include <iostream>
#include <vector>
#include "Image.h"

using namespace std;

template <typename T>
Image<T>::Image(vector<T> arg_pixels, int arg_order): pixels(arg_pixels), order(arg_order)
{
    length = arg_pixels.size();
}

template <typename T>
Image<T>::~Image()
{

}

template <typename T>
vector<T> Image<T>::get_pixels()
{
    return pixels;
}

template <typename T>
int Image<T>::get_order(){
    return order;
}

template <typename T>
int Image<T>::get_length()
{
    return length;
}

template class Image<unsigned char>;
template class Image<unsigned short>;