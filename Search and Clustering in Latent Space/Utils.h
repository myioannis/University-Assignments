#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cstring>
#include <string>
#include <cmath>

#include "Image.h"

template <typename T>
using Distance_Function = double (*)(Image<T>* d1, Image<T>* d2);

long int modular_pow(long long int base, int exponent, long long int modulus);
long int mod(long int a, long int b);
template <typename T>
double manhattan_Distance(Image<T>* image1, Image<T>* image2);
int check_Endianness();
void swap_Endians(unsigned int& value);

template <typename T>
void swap_Endians_Reduced(T& value);
#endif