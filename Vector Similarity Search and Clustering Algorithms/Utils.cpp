#include "Utils.h"

using namespace std;

/* It implements modular exponention (used when powering to very high powers) */
long int modular_pow(long long int base, int exponent, long long int modulus)
{
    long int c = 1; // result
    
    base = base % modulus;  // update base if it is more than or equal to modulus
    if (base == 0)      // base is divisible by modulus
        return 0;

    while (exponent > 0)
    {
        if (exponent & 1)   // if exponent is odd, multiply base with c 
            c = (c*base) % modulus; 

        exponent = exponent >> 1;   // exponent must be even now 
        base = (base*base) % modulus;
    }

    return c;
}

/* It implements the modulo operation using bitwise operations (works with negative numbers as well) */
long int mod(long int a, long int b)
{
    return a & (b - 1);
}

/* Calculates the manhattan distance between two images */
double manhattan_Distance(Image* image1, Image* image2)
{
    if (image1->get_length() != image2->get_length())
    {
        cout << "Error: different image lengths" << endl;
        return -1;
    }

    double sum = 0;
    vector<unsigned char> pixels1 = image1->get_pixels();
    vector<unsigned char> pixels2 = image2->get_pixels();
    for (int i = 0; i < image1->get_length(); i++)
    {
        sum += abs(pixels1[i] - pixels2[i]);
    }
    return sum;
}

/* Checks the endianness of the system, because the file's contents are saved in High Endian architecture,
   so some of the contents (e.g. number of images, dimensions) might have to be converted depending on
   each computer's architecture (single bytes are not affected, so they're not converted) */
int check_Endianness(){
    unsigned int i = 1;  
    char *c = (char*)&i;  
    if (*c) return 0; //Little(Low) endian
    else return 1; //Big(High) endian - the file's format
}

/* It converts a High Endian integer to Low Endian and vice versa */
void swap_Endians(unsigned int& value)  
{  
    int leftmost_byte, left_middle_byle, right_middle_byte, rightmost_byte, result;
    leftmost_byte = (value & 0x000000FF) >> 0;  
    left_middle_byle = (value & 0x0000FF00) >> 8;  
    right_middle_byte = (value & 0x00FF0000) >> 16;  
    rightmost_byte = (value & 0xFF000000) >> 24;  
    leftmost_byte <<= 24;  
    left_middle_byle <<= 16;  
    right_middle_byte <<= 8;  
    rightmost_byte <<= 0;  
    value = (leftmost_byte | left_middle_byle | 
              right_middle_byte | rightmost_byte);  
}  