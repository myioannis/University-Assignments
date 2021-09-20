#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include "Utils.h"
#include "Input.h"
#include "LSH.h"
#include "Hypercube.h"
#include "Exact.h"
#include "Image.h"
#include "Cluster.h"

using namespace std;

Input::Input(int argc, char* argv[]): complete(0){
    //Default parameters
    #if LSH_FLAG //If we're on ./lsh executable or running the Cluster executable with the LSH method
        k=4; L=5; N=1; R=10000;
        parse_CommandLineArguments(argc,argv);
        read_Dataset();
        read_Queryset();
    #elif HYPERCUBE_FLAG //If we're on ./cube executable or running the Cluster executable with the HYPERCUBE method
        k=14; M=10; probes=2; N=1; R=10000;
        parse_CommandLineArguments(argc,argv);
        read_Dataset();
        read_Queryset();
    #elif CLUSTER_FLAG //If we're on ./cluster executable
        parse_CommandLineArguments(argc,argv);
        if (method=="LSH"){ k=4; L=3; R=900; } //If CLUSTER is called with LSH method
        else if (method=="Hypercube"){ k=14; M=10; probes=2; R=900; } //If CLUSTER is called with HYPERCUBE method
        else if (method=="Classic"){ } //If CLUSTER is called with CLASSIC method
        read_Dataset();
        read_Configuration();
    #endif
}

Input::~Input(){
    for (int i=0; i<Data.size(); i++){
        delete Data[i];
    }
    for (int i=0; i<Queries.size(); i++){
        delete Queries[i];
    }
}

void Input::parse_CommandLineArguments(int argc, char* argv[]){
    /*Members k,L,N,r are initialized to their default values in the constructor and they're only
      modified if the user has given other values*/
    for (int i = 1; i < argc; i++)
    {
        if ( (strcmp(argv[i], "-d") == 0) || (strcmp(argv[i], "-i") == 0) ) datasetPath = argv[i+1];
        else if (strcmp(argv[i], "-q") == 0) querysetPath = argv[i+1];
        else if (strcmp(argv[i], "-k") == 0) k = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-L") == 0) L = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-M") == 0) M = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-probes") == 0) probes = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-o") == 0) outputPath = argv[i+1];
        else if (strcmp(argv[i], "-N") == 0) N = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-R") == 0) R = atof(argv[i+1]);
        else if (strcmp(argv[i], "-m") == 0) method = argv[i+1];
        else if (strcmp(argv[i], "-c") == 0) configPath = argv[i+1];
        else if (strcmp(argv[i], "-complete") == 0)  complete = true;
    }
    // If any of these filenames was not given, we cannot proceed
    if (datasetPath.empty() || outputPath.empty()) {
        cout << "Missing dataset and/or output path arguments" << endl;
        exit(EXIT_FAILURE);
    }
    #if CLUSTER_FLAG
        if (configPath.empty() || method.empty()){
            cout << "Missing configuration path and/or method arguments" << endl;
            exit(EXIT_FAILURE);            
        }
    #else
        if (querysetPath.empty()){
            cout << "Missing queryset path arguments" << endl;
            exit(EXIT_FAILURE);     
        }
    #endif
}

void Input::read_Dataset(){
    FILE *datasetFile;
    if ((datasetFile = fopen(datasetPath.c_str(),"rb")) == NULL){
        cout << "Error1! opening dataset file" << endl;
        exit(1);
    }

    unsigned int magicNumber, numOfImages, numOfRows, numOfColumns;
    fread(&magicNumber, sizeof(unsigned int), 1, datasetFile); //magicNumber is not needed
    fread(&numOfImages, sizeof(unsigned int), 1, datasetFile);
    fread(&numOfRows, sizeof(unsigned int), 1, datasetFile);
    fread(&numOfColumns, sizeof(unsigned int), 1, datasetFile);
    // If the computer's Endianness doesn't match the file's, then we convert the integers to the other form
    if (!check_Endianness()) {swap_Endians(numOfImages);swap_Endians(numOfRows);swap_Endians(numOfColumns);}
    // Read the images (vectors)
    for(int image_num=0; image_num<numOfImages; image_num++){
        vector<unsigned char> pixels;
        unsigned char pixel;
        // reserve the capacity needed for an image in order to avoid extensive reallocations
        pixels.reserve(numOfRows*numOfColumns);
        for (int pixel_num=0; pixel_num<numOfRows*numOfColumns; pixel_num++){
            fread(&pixel, sizeof(unsigned char), 1, datasetFile);
            pixels.push_back(pixel);
        }
        Image* newImage = new Image(pixels,image_num);
        Data.push_back(newImage); // Push this image to the Dataset
    }
    fclose(datasetFile); 
}

void Input::read_Queryset(){
    FILE *querysetFile;
    if ((querysetFile = fopen(querysetPath.c_str(),"rb")) == NULL){
        cout << "Error! opening queryset file" << endl;
        exit(1);
    }

    unsigned int magicNumber, numOfImages, numOfRows, numOfColumns;
    fread(&magicNumber, sizeof(unsigned int), 1, querysetFile); //magicNumber is not needed
    fread(&numOfImages, sizeof(unsigned int), 1, querysetFile);
    fread(&numOfRows, sizeof(unsigned int), 1, querysetFile);
    fread(&numOfColumns, sizeof(unsigned int), 1, querysetFile);
    // If the computer's Endianness doesn't match the file's, then we convert the integers to the other form
    if (!check_Endianness()) {swap_Endians(numOfImages);swap_Endians(numOfRows);swap_Endians(numOfColumns);}
    // Read the images (vectors)
    for(int image_num=0; image_num<10 /*numOfImages*/; image_num++){
        vector<unsigned char> pixels;
        unsigned char pixel;
        // reserve the capacity needed for an image in order to avoid extensive reallocations
        pixels.reserve(numOfRows*numOfColumns);
        for (int pixel_num=0; pixel_num<numOfRows*numOfColumns; pixel_num++){
            fread(&pixel, sizeof(unsigned char), 1, querysetFile);
            pixels.push_back(pixel);
        }
        if (image_num==1){d = pixels.size();}
        Image* newImage = new Image(pixels,image_num);
        Queries.push_back(newImage); // Push this image to the Queryset
    }
    fclose(querysetFile); 
}

void Input::read_Configuration(){
    fstream configFile;
    configFile.open(configPath,ios::in);
    if (configFile.is_open()){
        string line;
        while(getline(configFile, line)){ 
            char* cline = new char [line.length()+1];
            strcpy(cline, line.c_str());
            char* parameter = strtok(cline,":");
            int value = atoi(strtok(NULL," "));
            if (strcmp(parameter,"number_of_clusters")==0) K=value;
            else if (strcmp(parameter,"number_of_vector_hash_tables")==0) L=value;
            else if (method=="LSH" && strcmp(parameter,"number_of_vector_hash_functions")==0) k=value;
            else if (strcmp(parameter,"max_number_M_hypercube")==0) M=value;
            else if (method=="LSH" && strcmp(parameter,"number_of_hypercube_dimensions")==0) k=value;
            else if (strcmp(parameter,"number_of_probes")==0) probes=value;
            delete[] cline;
        }
        configFile.close();
    }
}

void Input::new_Queryset(string newQuerysetPath){
    querysetPath = newQuerysetPath;
    for (int i=0; i<Queries.size(); i++){ //Deallocate the query images from the heap
        delete Queries[i];
    }
    Queries.clear(); //Removes all elements from the vector so that it gets filled with the new Queries
    read_Queryset();
}

void Input::find_Neighbors(Distance_Function distance_Function){
    ofstream output;
    output.open(outputPath);

    Exact myExact(k, N, R, Data, Queries);
    vector<vector<distancePair*>> ExactNearestNeighbors = myExact.calculate_N_Nearest(distance_Function);
    vector<chrono::duration<double>> tTrue = myExact.get_Times();
    
    Technique* myTechnique; //LSH OR HYPERCUBE
    #if LSH_FLAG
        myTechnique = new LSH(k,L,N,d,R,Data,Queries);
        method = "LSH";
    #elif HYPERCUBE_FLAG
        myTechnique = new Hypercube(k,M,N,d,probes,R,Data,Queries);
        method = "Hypercube";
    #endif
    vector<vector<distancePair*>> nearestNeighbors = myTechnique->calculate_N_Nearest(distance_Function);
    vector<vector<distancePair*>> rangeNeighbors = myTechnique->calculate_in_Range(distance_Function);
    vector<chrono::duration<double>> tTechnique = myTechnique->get_Times();
    int notfound = 0;
    double maxAF = -1.0;
    vector<double> averageAFs;
    double averageAF = 0.0;
    for (int query = 0; query < nearestNeighbors.size(); query++)
    {
        output << "Query: " << query+1 << endl;
        //if (nearestNeighbors[query].size() == 0) notfound++;
        for (int neighbor = 0; neighbor < nearestNeighbors[query].size(); neighbor++)
        {
            if (nearestNeighbors[query][neighbor]->first == numeric_limits<double>::max()) notfound++;
            output << "Nearest neighbor-" << neighbor+1 << ": " << nearestNeighbors[query][neighbor]->second->get_order() << endl;
            output << "distance" << method << ": " << nearestNeighbors[query][neighbor]->first << endl;
            output << "distanceTrue: " << ExactNearestNeighbors[neighbor][query]->first << endl;

            double af =  nearestNeighbors[query][neighbor]->first / ExactNearestNeighbors[neighbor][query]->first;
            if (af > maxAF) maxAF = af;
            averageAFs.push_back(af);
        }
        output << "t" << method << ": " << tTechnique[query].count() << endl;
        output << "tTrue: " << tTrue[query].count() << endl;

        output << "R_near neighbors:" << endl;
        for (int neighbor = 0; neighbor < rangeNeighbors[query].size(); neighbor++)
        {
            output << rangeNeighbors[query][neighbor]->second->get_order() << endl;
        }
        output << endl;
    }

    double meanTimeSearch = 0.0, meanTimeSearchBF = 0.0;
    for (int i = 0; i < tTechnique.size(); i++) {
        meanTimeSearch += tTechnique[i].count();
    }
    for (int i = 0; i < tTrue.size(); i++) {
        meanTimeSearchBF += tTrue[i].count();
    }
    for (int i = 0; i < averageAFs.size(); i++) {
        averageAF += averageAFs[i];
    }
    cout << "meanTimeSearch" << method << ": " << meanTimeSearch / tTechnique.size() << " seconds" << endl;
    cout << "meanTimeSearchBF: " << meanTimeSearchBF / tTrue.size() << " seconds" << endl;
    cout << "maxAF: " << maxAF << endl;
    cout << "averageAF: " << averageAF / averageAFs.size() << endl;
    cout << "not found: " << notfound << endl;

    delete myTechnique;
    output.close();
}

void Input::find_Clusters(Distance_Function distance_Function){
    ofstream output;
    output.open(outputPath);

    // Print the method
    output << "Algorithm: ";
    if (method=="Classic") output << "Lloyds" << endl;
    else if (method=="LSH") output << "Range Search LSH" << endl;
    else if (method=="Hypercube") output << "Range Search Hypercube" << endl;
    // Make the clusters
    auto cluster_start = chrono::high_resolution_clock::now();
    Cluster myCluster(K,w,L,k,d,M,probes,R,method,Data,distance_Function);
    vector<vector<Image*>> clusters = myCluster.make_Clusters();
    auto cluster_end = chrono::high_resolution_clock::now();
    auto cluster_duration = chrono::duration_cast<chrono::seconds>(cluster_end - cluster_start);
    //For each cluster, print its centroid and the centroid's coordinates
    for (int cluster_num=0; cluster_num<clusters.size(); cluster_num++){
        output<<"CLUSTER-"<<cluster_num<<" {size: "<<clusters[cluster_num].size()-1<<", centroid: [";
        for (int pixel_num=0; pixel_num<clusters[cluster_num][0]->get_length(); pixel_num++){
            //the first item/point (index 0) in the vector of points for each cluster is the Centroid
            output << +clusters[cluster_num][0]->get_pixels()[pixel_num] << ",";
        }
        output << "]" << endl;
    }
    output<<"clustering_time: "<< cluster_duration.count() << " seconds" << endl;
    // Compute the silhouette
    vector<double> s = myCluster.compute_Silhouette();
    // Calculate silhouette coefficient <for each cluster> (average s(i) over all i in some cluster)
    output << "Silhouette: [";
    for (int cluster_num=0; cluster_num<clusters.size(); cluster_num++){
        double cluster_sum = 0;
        for (int point_num=0; point_num<clusters[cluster_num].size(); point_num++){
            cluster_sum+=s[clusters[cluster_num][point_num]->get_order()];
        }
        output << "s" << cluster_num << "=" << cluster_sum/clusters[cluster_num].size() << ",";
    }
    // Calculate <overall> silhouette coefficient (average s(i) over all i in dataset)
    double overall_sum = 0; 
    for (int i=0; i<Data.size(); i++){
        overall_sum+=s[i];
    }
    output << "stotal=" << overall_sum/Data.size() << "]" << endl; 
    if (complete){
        //Print all the points in each cluster
        output << "-complete" << endl;
        for (int cluster_num=0; cluster_num<clusters.size(); cluster_num++){
            output<<"CLUSTER-"<<cluster_num<<" {";
            for (int point_num=0; point_num<clusters[cluster_num].size(); point_num++){
                output << clusters[cluster_num][point_num]->get_order() << ", ";
            }
            output << "}" << endl;
        }

    }

    output.close();
}