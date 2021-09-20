For more info about the specifics of the code please read the README.pdf file.

# Course

This is an assigment for the 'Software Development for Algorithmic Problems' course at the Department of Informatics and Telecommunications, University of Athens.

# Short Description

Implementation of vector similarity search and vector clustering algorithms in C/C++ and use of them on the MNIST dataset of handwritten digits.

# Featured

- **Vector Similarity Search**
  - Exact Nearest Neighbors
  - Approximate Nearest Neighbors
    - Locality-Sensitive Hashing
    - Randomized Projections

<br>

- **Vector Clustering**
    - K-Means (Centroid-based Clustering)
      - Initialization++ (i.e. K-Means++)
      - Classic (Loyd's) Algorithm
      - Assignment by Direct Method:
        - Exact Approach (i.e. for each datapoint return nearest centroid)
        - Approximate Approach (use LSH and return approximate nearest centroid)
      - Reverse Assignment:
        - Approximate Range Search using LSH or Randomized Projections
      - Cluster Quality Assessment using the Silhouette method
