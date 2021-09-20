#include <iostream>
#include <vector>
#include "Image.h"

using namespace std;

Image::Image(vector<unsigned char> arg_pixels, int arg_order): pixels(arg_pixels), order(arg_order)
{
    length = arg_pixels.size();
}

Image::~Image()
{

}

vector<unsigned char> Image::get_pixels()
{
    return pixels;
}

int Image::get_order(){
    return order;
}

int Image::get_length()
{
    return length;
}