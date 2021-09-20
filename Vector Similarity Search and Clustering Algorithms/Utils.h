#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <cstring>
#include <string>
#include <cmath>

#include "Image.h"

typedef double (*Distance_Function)(Image* d1, Image* d2);

long int modular_pow(long long int base, int exponent, long long int modulus);
long int mod(long int a, long int b);
double manhattan_Distance(Image* image1, Image* image2);
int check_Endianness();
void swap_Endians(unsigned int& value);
#endif