#ifndef INPUT_H
#define INPUT_H

#include <string>
#include <vector>
#include "Utils.h"
#include "Cluster.h"

template <typename T>
class Image;

template <typename T1, typename T2>
class Input{
    private:
        int w; //window
        int K; //Number of clusters
        int L; //Number of hashtables
        int k; //Number of buckets
        int d; //Dimensions/Number of pixels
        int reduced_d; //Dimensions/Number of pixels
        int N; //Number of Nearest Neighbors
        int M;
        int probes;
        double R; //Range for range_search
        vector<Image<T1>*> Data;
        vector<Image<T1>*> Queries;
        vector<Image<T2>*> reducedData;
        vector<Image<T2>*> reducedQueries;
        Cluster<T1>* nnCluster;
        string datasetPath;
        string querysetPath;
        string reducedDatasetPath;
        string reducedQuerysetPath;
        string dataLabelsPath;
        string queriesLabelsPath;
        string outputPath;
        string configPath;
        string clustersPath;
        string method; //LSH, Hypercube or Classic
        int numData;
        int numQueries;
        vector<T1> dataLabels;
        vector<T1> queriesLabels;
        bool EMD; //complete flag for clustering

        /* Parses the command line arguments and assigns the correct values to the class' members */
        void parse_CommandLineArguments(int argc, char* argv[]);
        /* Reads the dataset from the file, creates the images in the heap and assigns them to the Data member */
        void read_Dataset();
        /* Reads the reduced dimension dataset from the file, creates the images in the heap and assigns them to the reducedData member */
        void read_Reduced_Dataset();
        /* Reads the queryset from the file, creates the images in the heap and assigns them to the Queries member */
        void read_Queryset();
        /* Reads the reduced dimension queryset from the file, creates the images in the heap and assigns them to the reducedQueries member */
        void read_Reduced_Queryset();
        /* Reads the dataset and queryset labels from both files and assigns them to the dataLabels and queriesLabels members */
        void read_Labels();
        /* Reads the configuration file for the clustering and assigns the correct values to the class' members  */
        void read_Configuration();
        /* Reads the NN clusters from the file  */
        void read_NN_Clusters();
    public:
        Input(int argc, char* argv[]);
        ~Input();
        /* Reads a different queryset for further classification */
        void new_Queryset(string newQuerysetFilename);
        /* Finds the N nearest neighbors for each query + range search and writes the results to the output file */
        void find_Neighbors(Distance_Function<T1> distance_Function, Distance_Function<T2> reduced_distance_Function);
        /* Creates the clusters of the Data and writes the results to the output file */
        void find_Clusters(Distance_Function<T1> distance_Function, Distance_Function<T2> reduced_distance_Function);
};
#endif