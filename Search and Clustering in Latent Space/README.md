For more info about the specifics of the code please read the README.pdf file.

# Course

This is an assigment for the 'Software Development for Algorithmic Problems' course at the Department of Informatics and Telecommunications, University of Athens.

# Short Description

A combination of the assignments 'Vector Similarity Search and Clustering Algorithms' (1) and 'Autoencoder as a Classifier' (2). Use of the autoencoder in (2) to create latent representations (reduce the dimension of the data) of the images in the MNIST dataset. Use of the methods in (1) for search and clustering of the images in the new (latent) space and comparison with the initial space.

# Tech Stack

**C/C++**, **Python**, **Keras**, PULP Linear Programming (for EMD), Pandas, Numpy, Matplotlib

# Featured

- Exact and approximate Nearest Neighbors computation using the latent representations of the images. Comparison with the results using the initial representations of the images. 

<br>

- Implementation of the **Earth Mover’s Distance** as a distance metric between the images in python (using linear programming). Use of this metric for Nearest Neighbors computation and comparison with the other metrics as per the running time and correctness of each.

<br>

- K-Medians clustering of the data in the new (latent) space and the initial space. Use of the neural networks in assigment (2) to classify the data and extract the clusters based on the resulting classification. Use of the silhuette method to compare the different clustering methods.
