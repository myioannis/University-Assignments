#include <iostream>
#include "Input.h"
#include "Image.h"
#include "Utils.h"

using namespace std;

int main(int argc, char* argv[]){
    Input<unsigned char, unsigned short> myInput(argc,argv);
    #if defined(SEARCH_FLAG)
        myInput.find_Neighbors(manhattan_Distance, manhattan_Distance);
        string more;
        string querysetPath;
        while(true){
            cout << "Would you like do continue the classification with a different queryset? (yes/no)" << endl;
            cin >> more;
            if (more=="yes"){
                cout << "Please type the whole path of the new queryset" << endl;
                cin >> querysetPath;
                myInput.new_Queryset(querysetPath);
                myInput.find_Neighbors(manhattan_Distance, manhattan_Distance);
            }
            else break;
        }
    #elif defined(CLUSTER_FLAG)
        myInput.find_Clusters(manhattan_Distance, manhattan_Distance);
    #endif
}