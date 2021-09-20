#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include "Utils.h"
#include "Input.h"
#include "LSH.h"
#include "Hypercube.h"
#include "Exact.h"
#include "Image.h"

using namespace std;

template <typename T1, typename T2>
Input<T1, T2>::Input(int argc, char* argv[]): EMD(0){
    //Default parameters
    #if SEARCH_FLAG //If we're on ./search executable
        k=4; L=5; N=10; R=1;
        parse_CommandLineArguments(argc,argv);
        read_Dataset();
        read_Queryset();
        if (EMD)
            read_Labels();
        else{
            N=1;
            read_Reduced_Dataset();
            read_Reduced_Queryset();
        }
    #elif CLUSTER_FLAG //If we're on ./cluster executable
        parse_CommandLineArguments(argc,argv);
        read_Dataset();
        read_Reduced_Dataset();
        read_Configuration();
        read_NN_Clusters();
    #endif
}

template <typename T1, typename T2>
Input<T1, T2>::~Input(){
    #if CLUSTER_FLAG
        delete nnCluster;
    #endif
    for (int i=0; i<Data.size(); i++){
        delete Data[i];
    }
    for (int i=0; i<Queries.size(); i++){
        delete Queries[i];
    }
    for (int i=0; i<reducedData.size(); i++){
        delete reducedData[i];
    }
    for (int i=0; i<reducedQueries.size(); i++){
        delete reducedQueries[i];
    }
}

template <typename T1, typename T2>
void Input<T1, T2>::parse_CommandLineArguments(int argc, char* argv[]){
    /*Members k,L,N,r are initialized to their default values in the constructor and they're only
      modified if the user has given other values*/
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-d") == 0) datasetPath = argv[i+1];
        else if (strcmp(argv[i], "-i") == 0) reducedDatasetPath = argv[i+1];
        else if (strcmp(argv[i], "-q") == 0) querysetPath = argv[i+1];
        else if (strcmp(argv[i], "-s") == 0) reducedQuerysetPath = argv[i+1];
        else if (strcmp(argv[i], "-l1") == 0) dataLabelsPath = argv[i+1];
        else if (strcmp(argv[i], "-l2") == 0) queriesLabelsPath = argv[i+1];
        else if (strcmp(argv[i], "-k") == 0) k = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-L") == 0) L = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-M") == 0) M = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-probes") == 0) probes = atoi(argv[i+1]);
        else if (strcmp(argv[i], "-o") == 0) outputPath = argv[i+1];
        else if (strcmp(argv[i], "-c") == 0) configPath = argv[i+1];
        else if (strcmp(argv[i], "-n") == 0) clustersPath = argv[i+1];
        else if (strcmp(argv[i], "-EMD") == 0) EMD = true;
        method = "Classic";
    }
    // If any of these filenames was not given, we cannot proceed
    if (datasetPath.empty() || outputPath.empty()) {
        cout << "Missing dataset and/or output path arguments" << endl;
        exit(EXIT_FAILURE);
    }
    if (EMD && (dataLabelsPath.empty() || queriesLabelsPath.empty())) {
        cout << "Missing label path arguments" << endl;
        exit(EXIT_FAILURE);
    }
    #if CLUSTER_FLAG
        if (configPath.empty() || clustersPath.empty()){
            cout << "Missing configuration path and/or NN clusters path" << endl;
            exit(EXIT_FAILURE);            
        }
    #else
        if (querysetPath.empty()){
            cout << "Missing queryset path arguments" << endl;
            exit(EXIT_FAILURE);     
        }
        if (!EMD && (reducedDatasetPath.empty() || reducedQuerysetPath.empty())) {
            cout << "Missing reduced path arguments" << endl;
            exit(EXIT_FAILURE);
        }
    #endif
}

template <typename T1, typename T2>
void Input<T1, T2>::read_Dataset(){
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
    if (!check_Endianness()) {swap_Endians(magicNumber);swap_Endians(numOfImages);swap_Endians(numOfRows);swap_Endians(numOfColumns);}

    cout << "How many data images would you like to use out of " << numOfImages << ": " << endl;
    cin >> numData;

    // Read the images (vectors)
    for(int image_num=0; image_num<numData; image_num++){
        vector<T1> pixels;
        T1 pixel;
        // reserve the capacity needed for an image in order to avoid extensive reallocations
        pixels.reserve(numOfRows*numOfColumns);
        for (int pixel_num=0; pixel_num<numOfRows*numOfColumns; pixel_num++){
            fread(&pixel, sizeof(T1), 1, datasetFile);
            pixels.push_back(pixel);
        }
        Image<T1>* newImage = new Image<T1>(pixels,image_num);
        Data.push_back(newImage); // Push this image to the Dataset
    }

    fclose(datasetFile);
}

template <typename T1, typename T2>
void Input<T1, T2>::read_Reduced_Dataset(){
    FILE *reducedDatasetFile;
    if ((reducedDatasetFile = fopen(reducedDatasetPath.c_str(),"rb")) == NULL){
        cout << "Error2! opening dataset file" << endl;
        exit(1);
    }

    unsigned int magicNumber, numOfImages, numOfRows, numOfColumns;
    fread(&magicNumber, sizeof(unsigned int), 1, reducedDatasetFile); //magicNumber is not needed
    fread(&numOfImages, sizeof(unsigned int), 1, reducedDatasetFile);
    fread(&numOfRows, sizeof(unsigned int), 1, reducedDatasetFile);
    fread(&numOfColumns, sizeof(unsigned int), 1, reducedDatasetFile);
    // If the computer's Endianness doesn't match the file's, then we convert the integers to the other form
    if (!check_Endianness()) {swap_Endians(magicNumber);swap_Endians(numOfImages);swap_Endians(numOfRows);swap_Endians(numOfColumns);}
    // Read the images (vectors)
    for(int image_num=0; image_num<numData; image_num++){
        vector<T2> pixels;
        T2 pixel;
        // reserve the capacity needed for an image in order to avoid extensive reallocations
        pixels.reserve(numOfRows*numOfColumns);
        for (int pixel_num=0; pixel_num<numOfRows*numOfColumns; pixel_num++){
            fread(&pixel, sizeof(T2), 1, reducedDatasetFile);
            if (!check_Endianness()) {swap_Endians_Reduced(pixel);}
            pixels.push_back(pixel);
        }
        Image<T2>* newImage = new Image<T2>(pixels,image_num);
        reducedData.push_back(newImage); // Push this image to the Dataset
    }

    fclose(reducedDatasetFile);
}

template <typename T1, typename T2>
void Input<T1, T2>::read_Queryset(){
    FILE *querysetFile;
    if ((querysetFile = fopen(querysetPath.c_str(),"rb")) == NULL){
        cout << "Error3! opening queryset file" << endl;
        exit(1);
    }

    unsigned int magicNumber, numOfImages, numOfRows, numOfColumns;
    fread(&magicNumber, sizeof(unsigned int), 1, querysetFile); //magicNumber is not needed
    fread(&numOfImages, sizeof(unsigned int), 1, querysetFile);
    fread(&numOfRows, sizeof(unsigned int), 1, querysetFile);
    fread(&numOfColumns, sizeof(unsigned int), 1, querysetFile);
    // If the computer's Endianness doesn't match the file's, then we convert the integers to the other form
    if (!check_Endianness()) {swap_Endians(numOfImages);swap_Endians(numOfRows);swap_Endians(numOfColumns);}

    cout << "How many query images would you like to use out of " << numOfImages << ": " << endl;
    cin >> numQueries;

    // Read the images (vectors)
    for(int image_num=0; image_num<numQueries; image_num++){
        vector<T1> pixels;
        T1 pixel;
        // reserve the capacity needed for an image in order to avoid extensive reallocations
        pixels.reserve(numOfRows*numOfColumns);
        for (int pixel_num=0; pixel_num<numOfRows*numOfColumns; pixel_num++){
            fread(&pixel, sizeof(T1), 1, querysetFile);
            pixels.push_back(pixel);
        }
        Image<T1>* newImage = new Image<T1>(pixels,image_num);
        if (image_num==1){d = pixels.size();}
        Queries.push_back(newImage); // Push this image to the Queryset
    }
    fclose(querysetFile);
}

template <typename T1, typename T2>
void Input<T1, T2>::read_Reduced_Queryset(){
    FILE *reducedQuerysetFile;
    if ((reducedQuerysetFile = fopen(reducedQuerysetPath.c_str(),"rb")) == NULL){
        cout << "Error4! opening queryset file" << endl;
        exit(1);
    }

    unsigned int magicNumber, numOfImages, numOfRows, numOfColumns;
    fread(&magicNumber, sizeof(unsigned int), 1, reducedQuerysetFile); //magicNumber is not needed
    fread(&numOfImages, sizeof(unsigned int), 1, reducedQuerysetFile);
    fread(&numOfRows, sizeof(unsigned int), 1, reducedQuerysetFile);
    fread(&numOfColumns, sizeof(unsigned int), 1, reducedQuerysetFile);
    // If the computer's Endianness doesn't match the file's, then we convert the integers to the other form
    if (!check_Endianness()) {swap_Endians(numOfImages);swap_Endians(numOfRows);swap_Endians(numOfColumns);}
    // Read the images (vectors)
    for(int image_num=0; image_num<numQueries; image_num++){
        vector<T2> pixels;
        T2 pixel;
        // reserve the capacity needed for an image in order to avoid extensive reallocations
        pixels.reserve(numOfRows*numOfColumns);
        for (int pixel_num=0; pixel_num<numOfRows*numOfColumns; pixel_num++){
            fread(&pixel, sizeof(T2), 1, reducedQuerysetFile);
            if (!check_Endianness()) swap_Endians_Reduced(pixel);
            pixels.push_back(pixel);
        }
        Image<T2>* newImage = new Image<T2>(pixels,image_num);
        if (image_num==1){reduced_d = pixels.size();}
        reducedQueries.push_back(newImage); // Push this image to the Queryset
    }

    fclose(reducedQuerysetFile); 
}

template <typename T1, typename T2>
void Input<T1, T2>::read_Labels(){
    FILE *dataLabelsFile, *queriesLabelsFile;
    if ((dataLabelsFile = fopen(dataLabelsPath.c_str(),"rb")) == NULL){
        cout << "Error11! opening dataLabels file" << endl;
        exit(1);
    }
    if ((queriesLabelsFile = fopen(queriesLabelsPath.c_str(),"rb")) == NULL){
        cout << "Error111! opening queriesLabels file" << endl;
        exit(1);
    }

    unsigned int magicNumber, numOfImages;
    fread(&magicNumber, sizeof(unsigned int), 1, dataLabelsFile); //magicNumber is not needed
    fread(&numOfImages, sizeof(unsigned int), 1, dataLabelsFile);
    // If the computer's Endianness doesn't match the file's, then we convert the integers to the other form
    if (!check_Endianness()) {swap_Endians(magicNumber);swap_Endians(numOfImages);}

    dataLabels.reserve(numData);
    // Read the images (vectors)
    for(int image_num=0; image_num<numData; image_num++){
        T1 label;
        fread(&label, sizeof(T1), 1, dataLabelsFile);
        dataLabels.push_back(label); // Push this image to the Dataset
    }

    fread(&magicNumber, sizeof(unsigned int), 1, queriesLabelsFile); //magicNumber is not needed
    fread(&numOfImages, sizeof(unsigned int), 1, queriesLabelsFile);
    // If the computer's Endianness doesn't match the file's, then we convert the integers to the other form
    if (!check_Endianness()) {swap_Endians(magicNumber);swap_Endians(numOfImages);}
    queriesLabels.reserve(numQueries);
    // Read the images (vectors)
    for(int image_num=0; image_num<numQueries; image_num++){
        T1 label;
        fread(&label, sizeof(T1), 1, queriesLabelsFile);
        queriesLabels.push_back(label); // Push this image to the Dataset
    }

    fclose(dataLabelsFile);
    fclose(queriesLabelsFile);
}

template <typename T1, typename T2>
void Input<T1, T2>::read_Configuration(){
    fstream configFile;
    configFile.open(configPath,ios::in);
    if (configFile.is_open()){
        string line;
        while(getline(configFile, line)){ 
            char* cline = new char [line.length()+1];
            strcpy(cline, line.c_str());
            if (strcmp(cline, "") != 0 && strcmp(cline, " ") != 0 && strcmp(cline, "\n") != 0 && strcmp(cline, "\t") != 0) {
                char* parameter = strtok(cline,":");
                if (strcmp(parameter,"number_of_clusters")==0) K=atoi(strtok(NULL," "));
                else if (strcmp(parameter,"number_of_vector_hash_tables")==0) L=atoi(strtok(NULL," "));
                else if (strcmp(parameter,"number_of_vector_hash_functions")==0) k=atoi(strtok(NULL," "));
            }
            delete[] cline;
        }
        configFile.close();
    }
}

template <typename T1, typename T2>
void Input<T1, T2>::read_NN_Clusters(){
    fstream clustersFile;
    clustersFile.open(clustersPath, ios::in);

    if (clustersFile.is_open()){
        // Create a cluster class for the neural network precalculated classification (K classes/clusters)
        nnCluster = new Cluster<T1>(K, Data, manhattan_Distance);
        vector<vector<Image<T1>*>> clusters;
        clusters.reserve(K);

        string line;
        int cluster_num, size;
        while(getline(clustersFile, line)){
            char* cline = new char [line.length()+1];
            strcpy(cline, line.c_str());

            // Read each cluster info from clusters file
            sscanf(cline, "CLUSTER-%d { size: %d", &cluster_num, &size);
            vector<Image<T1>*> cluster;
            cluster.reserve(size);

            char* cluster_info = strtok(cline,",");
            char* img_info = strtok(NULL," ");
            Image<T1>* img;
            char img_str[10];
            int img_num;

            // Read all images from file and assign them to current cluster
            while (img_info[strlen(img_info)-1] == ','){
                memset(img_str, 0, 10);
                strncpy(img_str, img_info, strlen(img_info)-1);
                img_num = atoi(img_str);

                // Add only the images that are also in the dataset
                if (img_num < Data.size()) {
                    img = Data[img_num];
                    cluster.push_back(Data[img_num]);
                }

                img_info = strtok(NULL," ");
            }

            memset(img_str, 0, 10);
            strncpy(img_str, img_info, strlen(img_info)-1);
            img_num = atoi(img_str);

            if (img_num < Data.size()) {
                img = Data[img_num];
                cluster.push_back(Data[img_num]);
            }

            clusters.push_back(cluster);

            delete[] cline;
        }
        nnCluster->new_Centroids(clusters);
        nnCluster->set_Clusters(clusters);
        clustersFile.close();
    }
}

template <typename T1, typename T2>
void Input<T1, T2>::new_Queryset(string nQuerysetPath){
    querysetPath = nQuerysetPath;
    for (int i=0; i<Queries.size(); i++){ //Deallocate the query images from the heap
        delete Queries[i];
    }
    Queries.clear(); //Removes all elements from the vector so that it gets filled with the new Queries
    read_Queryset();
}

template <typename T1, typename T2>
void Input<T1, T2>::find_Neighbors(Distance_Function<T1> distance_Function, Distance_Function<T2> reduced_distance_Function){
    Exact<T1> myExact(k, N, R, Data, Queries);
    vector<vector<distancePair<T1>*>> ExactNearestNeighbors = myExact.calculate_N_Nearest(distance_Function);
    vector<chrono::duration<double>> tTrue = myExact.get_Times();

    if (EMD){
        string command = "python emd.py -d " + datasetPath + ".gz -q " + querysetPath + ".gz -l1 " + dataLabelsPath + ".gz -l2 " + queriesLabelsPath + ".gz -o " + outputPath + " -nd " + to_string(numData) + " -nq " + to_string(numQueries);
        system(command.c_str());

        ofstream output;
        output.open(outputPath, ofstream::app);

        int total = 0;
        for (int query=0; query<numQueries; query++){
            for (int neighbor = 0; neighbor < N; neighbor++){
                if (queriesLabels[query] == dataLabels[ExactNearestNeighbors[neighbor][query]->second->get_order()]){
                    total++;
                }
            }
        }
        double avg = (double)total/numQueries;

        double time_elapsed = 0.0;
        for (int i = 0; i < tTrue.size(); i++)
            time_elapsed += tTrue[i].count();

        output << "Average Correct Search Results MANHATTAN: " << avg << "/" << N << endl;
        output << "Minutes Elapsed MANHATTAN: " << time_elapsed/60 << endl;
        output.close();
    }
    else{
        ofstream output;
        output.open(outputPath);

        Technique<T1>* myTechnique = new LSH<T1>(k,L,N,d,R,Data,Queries);
        method = "LSH";

        vector<vector<distancePair<T1>*>> nearestNeighbors = myTechnique->calculate_N_Nearest(distance_Function);
        vector<chrono::duration<double>> tTechnique = myTechnique->get_Times();

        Exact<T2> myReduced(k, N, R, reducedData, reducedQueries);
        vector<vector<distancePair<T2>*>> ReducedNearestNeighbors = myReduced.calculate_N_Nearest(reduced_distance_Function);
        vector<chrono::duration<double>> tReduced = myReduced.get_Times();

        int notfound = 0;
        double maxAF = -1.0;
        vector<double> averageAFs;
        vector<double> reducedAFs;
        double averageAF = 0.0;
        double reducedAF = 0.0;
        for (int query = 0; query < nearestNeighbors.size(); query++)
        {
            output << "Query: " << query+1 << endl;
            //if (nearestNeighbors[query].size() == 0) notfound++;
            for (int neighbor = 0; neighbor < nearestNeighbors[query].size(); neighbor++)
            {
                if (nearestNeighbors[query][neighbor]->first == numeric_limits<double>::max()) notfound++;
                double distanceReduced = distance_Function(Queries[query], Data[ReducedNearestNeighbors[neighbor][query]->second->get_order()]);
                output << "Nearest neighbor Reduced: " << ReducedNearestNeighbors[neighbor][query]->second->get_order() << endl;
                output << "Nearest neighbor LSH: " << nearestNeighbors[query][neighbor]->second->get_order() << endl;
                output << "Nearest neighbor True: " << ExactNearestNeighbors[neighbor][query]->second->get_order() << endl;
                output << "distanceReduced: " << distanceReduced << endl;
                output << "distanceLSH: " << nearestNeighbors[query][neighbor]->first << endl;
                output << "distanceTrue: " << ExactNearestNeighbors[neighbor][query]->first << endl;

                double af = nearestNeighbors[query][neighbor]->first / ExactNearestNeighbors[neighbor][query]->first;
                double reduced_af = distanceReduced / ExactNearestNeighbors[neighbor][query]->first;
                if (reduced_af > maxAF) maxAF = reduced_af;
                averageAFs.push_back(af);
                reducedAFs.push_back(reduced_af);
            }
            output << endl;
        }

        double meanTimeSearchReduced = 0.0, meanTimeSearch = 0.0, meanTimeSearchBF = 0.0;
        for (int i = 0; i < tTechnique.size(); i++) {
            meanTimeSearchReduced += tReduced[i].count();
        }
        for (int i = 0; i < tTechnique.size(); i++) {
            meanTimeSearch += tTechnique[i].count();
        }
        for (int i = 0; i < tTrue.size(); i++) {
            meanTimeSearchBF += tTrue[i].count();
        }
        for (int i = 0; i < averageAFs.size(); i++) {
            averageAF += averageAFs[i];
        }
        for (int i = 0; i < reducedAFs.size(); i++) {
            reducedAF += reducedAFs[i];
        }
        output << "Average tReduced: " << meanTimeSearchReduced / tReduced.size() << " seconds" << endl;
        output << "Average tLSH: " << meanTimeSearch / tTechnique.size() << " seconds" << endl;
        output << "Average tTrue: " << meanTimeSearchBF / tTrue.size() << " seconds" << endl;

        output << "Approximation Factor LSH: " << averageAF / averageAFs.size() << endl;
        output << "Approximation Factor Reduced: " << reducedAF / reducedAFs.size() << endl;

        delete myTechnique;
        output.close();
    }
}

template <typename T1, typename T2>
void Input<T1, T2>::find_Clusters(Distance_Function<T1> distance_Function, Distance_Function<T2> reduced_distance_Function){
    ofstream output;
    output.open(outputPath);

    output << "NEW SPACE" << endl;
    // Make the clusters
    auto cluster_start = chrono::high_resolution_clock::now();
    Cluster<T2> newCluster(K,w,L,k,reduced_d,M,probes,R,method,reducedData,reduced_distance_Function);
    vector<vector<Image<T2>*>> new_clusters = newCluster.make_Clusters();
    auto cluster_end = chrono::high_resolution_clock::now();
    auto cluster_duration = chrono::duration_cast<chrono::seconds>(cluster_end - cluster_start);
    //For each cluster, print its centroid and the centroid's coordinates
    for (int cluster_num=0; cluster_num<new_clusters.size(); cluster_num++){
        output<<"CLUSTER-"<<cluster_num+1<<" {size: "<<new_clusters[cluster_num].size()-1<<", centroid: [";
        for (int pixel_num=0; pixel_num<new_clusters[cluster_num][0]->get_length(); pixel_num++){
            //the first item/point (index 0) in the vector of points for each cluster is the Centroid
            output << +new_clusters[cluster_num][0]->get_pixels()[pixel_num] << ",";
        }
        output << "]" << endl;
    }
    output<<"clustering_time: "<< cluster_duration.count() << " seconds" << endl;
    // Compute the silhouette
    vector<double> new_s = newCluster.compute_Silhouette_Reduced(Data, distance_Function);
    // Calculate silhouette coefficient <for each cluster> (average s(i) over all i in some cluster)
    output << "Silhouette: [";
    for (int cluster_num=0; cluster_num<new_clusters.size(); cluster_num++){
        double cluster_sum = 0;
        for (int point_num=0; point_num<new_clusters[cluster_num].size(); point_num++){
            cluster_sum+=new_s[new_clusters[cluster_num][point_num]->get_order()];
        }
        output << "s" << cluster_num << "=" << cluster_sum/new_clusters[cluster_num].size() << ",";
    }
    // Calculate <overall> silhouette coefficient (average s(i) over all i in dataset)
    double overall_reduced_sum = 0; 
    for (int i=0; i<reducedData.size(); i++){
        overall_reduced_sum+=new_s[i];
    }
    output << "stotal=" << overall_reduced_sum/reducedData.size() << "]" << endl;
    output << "Value of Objective Function: " << newCluster.compute_Objective_Function_Reduced(Data, distance_Function) << endl;

    output << endl << "ORIGINAL SPACE" << endl;
    // Make the clusters
    cluster_start = chrono::high_resolution_clock::now();
    Cluster<T1> originalCluster(K,w,L,k,d,M,probes,R,method,Data,distance_Function);
    vector<vector<Image<T1>*>> original_clusters = originalCluster.make_Clusters();
    cluster_end = chrono::high_resolution_clock::now();
    cluster_duration = chrono::duration_cast<chrono::seconds>(cluster_end - cluster_start);
    //For each cluster, print its centroid and the centroid's coordinates
    for (int cluster_num=0; cluster_num<original_clusters.size(); cluster_num++){
        output<<"CLUSTER-"<<cluster_num+1<<" {size: "<<original_clusters[cluster_num].size()-1<<", centroid: [";
        for (int pixel_num=0; pixel_num<original_clusters[cluster_num][0]->get_length(); pixel_num++){
            //the first item/point (index 0) in the vector of points for each cluster is the Centroid
            output << +original_clusters[cluster_num][0]->get_pixels()[pixel_num] << ",";
        }
        output << "]" << endl;
    }
    output<<"clustering_time: "<< cluster_duration.count() << " seconds" << endl;
    // Compute the silhouette
    vector<double> original_s = originalCluster.compute_Silhouette();
    // Calculate silhouette coefficient <for each cluster> (average s(i) over all i in some cluster)
    output << "Silhouette: [";
    for (int cluster_num=0; cluster_num<original_clusters.size(); cluster_num++){
        double cluster_sum = 0;
        for (int point_num=0; point_num<original_clusters[cluster_num].size(); point_num++){
            cluster_sum+=original_s[original_clusters[cluster_num][point_num]->get_order()];
        }
        output << "s" << cluster_num << "=" << cluster_sum/original_clusters[cluster_num].size() << ",";
    }
    // Calculate <overall> silhouette coefficient (average s(i) over all i in dataset)
    double overall_original_sum = 0; 
    for (int i=0; i<Data.size(); i++){
        overall_original_sum+=original_s[i];
    }
    output << "stotal=" << overall_original_sum/Data.size() << "]" << endl;
    output << "Value of Objective Function: " << originalCluster.compute_Objective_Function() << endl;

    output << endl << "CLASSES AS CLUSTERS" << endl;
    vector<vector<Image<T1>*>> nn_clusters = nnCluster->get_Clusters();
    // Compute the silhouette
    vector<double> nn_s = nnCluster->compute_Silhouette();
    // Calculate silhouette coefficient <for each cluster> (average s(i) over all i in some cluster)
    output << "Silhouette: [";
    for (int cluster_num=0; cluster_num<nn_clusters.size(); cluster_num++){
        double cluster_sum = 0;
        for (int point_num=0; point_num<nn_clusters[cluster_num].size(); point_num++){
            cluster_sum+=nn_s[nn_clusters[cluster_num][point_num]->get_order()];
        }
        output << "s" << cluster_num << "=" << cluster_sum/nn_clusters[cluster_num].size() << ",";
    }
    // Calculate <overall> silhouette coefficient (average s(i) over all i in dataset)
    double overall_nn_sum = 0; 
    for (int i=0; i<Data.size(); i++){
        overall_nn_sum+=nn_s[i];
    }
    output << "stotal=" << overall_nn_sum/Data.size() << "]" << endl;
    output << "Value of Objective Function: " << nnCluster->compute_Objective_Function() << endl;

    output.close();
}

template class Input<unsigned char, unsigned short>;
template class Input<unsigned char, unsigned char>;