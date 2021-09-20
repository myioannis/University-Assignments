#ifndef INPUT_H
#define INPUT_H

#include <string>
#include <vector>
#include "Utils.h"

class Image;

class Input{
    private:
        int w; //window
        int K; //Number of clusters
        int L; //Number of hashtables
        int k; //Number of buckets
        int d; //Dimensions/Number of pixels
        int N; //Number of Nearest Neighbors
        int M;
        int probes;
        double R; //Range for range_search
        vector<Image*> Data;
        vector<Image*> Queries;
        string datasetPath;
        string querysetPath;
        string outputPath;
        string configPath;
        string method; //LSH, Hypercube or Classic
        bool complete; //complete flag for clustering

        /* Parses the command line arguments and assigns the correct values to the class' members */
        void parse_CommandLineArguments(int argc, char* argv[]);
        /* Reads the dataset from the file, creates the images in the heap and assigns them to the Data member */
        void read_Dataset();
        /* Reads the queryset from the file, creates the images in the heap and assigns them to the Queries member */
        void read_Queryset();
        /* Reads the configuration file for the clustering and assigns the correct values to the class' members  */
        void read_Configuration();
    public:
        Input(int argc, char* argv[]);
        ~Input();
        /* Reads a different queryset for further classification */
        void new_Queryset(string newQuerysetFilename);
        /* Finds the N nearest neighbors for each query + range search and writes the results to the output file */
        void find_Neighbors(Distance_Function distance_Function);
        /* Creates the clusters of the Data and writes the results to the output file */
        void find_Clusters(Distance_Function distance_Function);
};
#endif