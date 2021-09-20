from utilities import read_Data, read_Labels, EMD_parse_CLA #from our utilities.py
from pulp import LpMinimize, LpProblem, LpStatus, lpSum, LpVariable, GLPK_CMD
from scipy.optimize import linprog
from sklearn.utils import shuffle
from math import ceil,sqrt
import numpy as np
import sys, os
import time

def euclidean_distance(centroid1,centroid2):
  """ Calculates the euclidean distance between two centroid pixels """
  return sqrt((centroid1.x-centroid2.x)**2 + (centroid1.y-centroid2.y)**2)

class CentroidPixel():
  """ Represents the centroid pixel of a cluster """
  def __init__(self,x,y):
    self.x = x
    self.y = y

class PixelCluster():
  """ Represents a cluster of an image """
  def __init__(self,pixels,x_start,x_end,y_start,y_end,totalWeight):
    self.x_start = x_start
    self.x_end = x_end
    self.y_start = y_start
    self.y_end = y_end

    # Calculate the weight of the cluster (sum of pixels inside it, normalized to [0,1])
    self.weight = sum( pixels[row][column] for row in range(x_start,x_end+1) for column in range(y_start,y_end+1) ) / totalWeight

    # Set the centroid of the cluster
    centroidRow = (x_start+x_end)//2
    centroidColumn = (y_start+y_end)//2
    self.centroid = CentroidPixel(centroidRow,centroidColumn)

  def ground_distance(self,consumerCluster,numPixels,distanceFunction):
    """
      Calculates the ground distance (using distanceFunction) between this pixel cluster (self)
      as a supplier and the argument pixel cluster as a consumer
    """
    supplierCluster = self
    return distanceFunction(supplierCluster.centroid,consumerCluster.centroid)

class Image():
  """ Represents an image """
  def __init__(self,image,rowsPerCluster,columnsPerCluster):
    self.totalRows = image.shape[0]
    self.totalColumns = image.shape[1]
    self.totalWeight = sum(image.flatten())
    self.totalPixels = image
    self.numPixels = self.totalRows*self.totalColumns

    # Make the pixel clusters of the image
    self.clusters = [ PixelCluster(self.totalPixels,curRow,curRow+rowsPerCluster-1,curColumn,curColumn+columnsPerCluster-1,self.totalWeight) for curRow in range(0,self.totalRows,rowsPerCluster) for curColumn in range(0,self.totalColumns,columnsPerCluster) ]
    self.numClusters = len(self.clusters)

  def EMD(self,consumerImage,distanceFunction): #this image is the supplier and the argument is the consumer
    """ 
      Calculates the EMD distance between this pixel cluster (self)
      as a supplier and the argument pixel cluster as a consumer
    """
    # Create the model
    problem = LpProblem("EMD", LpMinimize)
    supplierImage = self
    m = len(supplierImage.clusters) #number of clusters of supplier image (self)
    n = len(consumerImage.clusters) #number of clusters of consumer image

    # Calculate the ground distances (distances between the centroids) between each cluster of the "supplier" image and each cluster of the "consumer" image
    distances = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            distances[i][j] = supplierImage.clusters[i].ground_distance(consumerImage.clusters[j],self.numPixels,distanceFunction)

    # Set the variables for EMD calculations (n*m=n^2 variables in total)
    variablesList = []
    for i in range(m):
        tempList = []
        for j in range(n):
            tempList.append(LpVariable("x"+str(i)+" "+str(j), lowBound = 0))
        variablesList.append(tempList)

    # Set the objective function (minimization of total work)
    objectiveFunction = []
    objectiveFunction = [ variablesList[i][j] * distances[i][j] for i in range(m) for j in range(n) ]
    problem += lpSum(objectiveFunction)

    # Set the Constraints
    # Constraint 1: allow moving supplies from supplierImage to consumerImage and not vice versa
    for i in range(m):
        for j in range(n):
            problem += (variablesList[i][j] >= 0)
    # Constraint 2: limit the amount of supplies that can be sent by the clusters in the supplierImage to their weights
    for i in range(m):
        constraint2 = [variablesList[i][j] for j in range(n)]
        problem += (lpSum(constraint2) == supplierImage.clusters[i].weight)
    # Constraint 3: limit the clusters in the consumerImage to receive no more supplies than their weights
    for j in range(n):
        constraint3 = [variablesList[i][j] for i in range(m)]
        problem += (lpSum(constraint3) == consumerImage.clusters[j].weight)
    # # Constraint 4: force to move the maximum amount of supplies possible (this would be used if the previous 2 constraints where inequalities <=)
    # constraint4 = [ variablesList[i][j] for i in range(m) for j in range(n) ]
    # problem += (lpSum(constraint4) == min(supplierImage.totalWeight, consumerImage.totalWeight))

    # Solve the problem
    status = problem.solve(GLPK_CMD(msg=False)) #timeLimit,gapRel
    flow = problem.objective.value()

    # Because of constraint (4), this factor is needed when the two signatures have different total weight, in order to avoid favoring smaller signatures
    #return flow / min(supplierImage.totalWeight, consumerImage.totalWeight)
    return flow

def find_divisors(n):
  """ Finds all divisors of a number, in order to know the possible divisions into clusters for the images """
  divs = {1,n}
  for i in range(2,int(sqrt(n))+1):
      if n%i == 0:
          divs.update((i,n//i))
  return list(sorted(divs))

if __name__ == '__main__':
    # Parse the command line arguments
    data_Path, queries_Path, dataLabels_Path, queryLabels_Path, output_Path, numData, numQueries = EMD_parse_CLA(sys.argv)
    # If any of the path arguments was not given (forgotten or running as a jupyter notebook), then ask for them
    if not data_Path: data_Path = input('Please provide the path of the training data:')
    if not queries_Path: queries_Path = input('Please provide the path of the test data:')
    if not dataLabels_Path: dataLabels_Path = input('Please provide the path of the training labels:')
    if not queryLabels_Path: queryLabels_Path = input('Please provide the path of the test labels:')
    if not output_Path: output_Path = input('Please provide the path of the output file:')

    # Read the all the datasets
    data = read_Data(data_Path) #train-images-idx3-ubyte.gz
    queries = read_Data(queries_Path) #t10k-images-idx3-ubyte.gz
    dataLabels = read_Labels(dataLabels_Path) #train-labels-idx1-ubyte.gz
    queryLabels = read_Labels(queryLabels_Path) #t10k-labels-idx1-ubyte.gz
    if not numData: numData = int(input(f"How many data images would you like to use out of {len(data)}\n"))
    if not numQueries: numQueries = int(input(f"How many query images would you like to use out of {len(queries)}\n"))

    # Read the number of clusters that each image will be partitioned into
    print("Into how many pixel clusters would you like to partition each image? Please choose one of the following:")
    divisors = find_divisors(data.shape[1])
    for i,divisor in enumerate(divisors):
      print(f'{i}) {int((data.shape[1]*data.shape[2])/(divisor*divisor))} clusters of {divisor}x{divisor} pixels each')
    choice = int(input(""))

    # Open the output file
    outputFile = open(output_Path, "w") 

    # Partition the train images into clusters
    #data, dataLabels = shuffle(data,dataLabels)
    dataImages = [ Image(image,divisors[choice],divisors[choice]) for image in data[:numData] ]

    # # Partition the test images into clusters
    #queries, queryLabels = shuffle(queries,queryLabels)
    queryImages = [ Image(image,divisors[choice],divisors[choice]) for image in queries[:numQueries] ]

    N = 10
    total = 0
    start = time.perf_counter()
    for i,query in enumerate(queryImages):
      distances_all = [ (query.EMD(datum,euclidean_distance),index) for index,datum in enumerate(dataImages) ]
      distances_nearest = sorted(distances_all)[:N]
      sameLabel_all = [ tup for tup in distances_all if dataLabels[tup[1]] == queryLabels[i] ]
      sameLabel_nearest = [ tup for tup in distances_nearest if dataLabels[tup[1]] == queryLabels[i] ]
      outputFile.write(f'Query {i} (label {queryLabels[i]}) has {len(sameLabel_all)} images with the same label (in the first {len(dataImages)} data), of which {len(sameLabel_nearest)} are in the {N} nearest neighbors!\n')
      total += len(sameLabel_nearest)
    outputFile.write(f"Average Correct Search Results EMD: {total/len(queryImages)}/{N}\n")
    outputFile.write(f"Minutes Elapsed EMD: {(time.perf_counter()-start)/60}\n")

    outputFile.close()

