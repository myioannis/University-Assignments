import gzip
import sys
import numpy as np

BYTEORDER_ = 'big'

def read_Data(filename):
  with gzip.open(filename) as bytestream:
    #   BYTEORDER_ = sys.byteorder
    #   if 'google.colab' in sys.modules: BYTEORDER_ = 'big'
    #   print(BYTEORDER_)
      magicNumber = int.from_bytes(bytestream.read(4),byteorder=BYTEORDER_,signed=False) #not used
      numOfImages = int.from_bytes(bytestream.read(4),byteorder=BYTEORDER_,signed=False)
      numOfRows = int.from_bytes(bytestream.read(4),byteorder=BYTEORDER_,signed=False)
      numOfColumns = int.from_bytes(bytestream.read(4),byteorder=BYTEORDER_,signed=False)
      buf = bytestream.read(numOfRows * numOfColumns * numOfImages)
      data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
      data = data.reshape(numOfImages, numOfRows,numOfColumns)
      return data

def read_Labels(filename):
  with gzip.open(filename) as bytestream:
    #   BYTEORDER_ = sys.byteorder
    #   if 'google.colab' in sys.modules: BYTEORDER_ = 'big'
    #   print(BYTEORDER_)
      magicNumber = int.from_bytes(bytestream.read(4),byteorder=BYTEORDER_,signed=False) #not used
      numOfImages = int.from_bytes(bytestream.read(4),byteorder=BYTEORDER_,signed=False)
      buf = bytestream.read(1 * numOfImages)
      labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
      return labels

def AE_parse_CLA(argv):
  trainData_Path = ''
  for index, argument in enumerate(argv):
    if argument == '-d': trainData_Path = argv[index+1]
  return trainData_Path

def CL_parse_CLA(argv):
  trainData_Path = ''
  testData_Path = ''
  trainLabels_Path = ''
  testLabels_Path = ''
  model_Path = ''  
  for index, argument in enumerate(argv):
    if argument == '-d': trainData_Path = argv[index+1]
    elif argument == '-dl': trainLabels_Path = argv[index+1]
    elif argument == '-t': testData_Path = argv[index+1]
    elif argument == '-tl': testLabels_Path = argv[index+1]
    elif argument == '-model': model_Path = argv[index+1]
  return trainData_Path,testData_Path,trainLabels_Path,testLabels_Path,model_Path

def EMD_parse_CLA(argv):
  data_Path = ''
  queries_Path = ''
  dataLabels_Path = ''
  queryLabels_Path = ''
  output_Path = ''  
  numData = 0
  numQueries = 0
  for index, argument in enumerate(argv):
    if argument == '-d': data_Path = argv[index+1]
    elif argument == '-l1': dataLabels_Path = argv[index+1]
    elif argument == '-q': queries_Path = argv[index+1]
    elif argument == '-l2': queryLabels_Path = argv[index+1]
    elif argument == '-o': output_Path = argv[index+1]
    elif argument == '-nd': numData = argv[index+1]
    elif argument == '-nq': numQueries = argv[index+1]
  return data_Path,queries_Path,dataLabels_Path,queryLabels_Path,output_Path,int(numData),int(numQueries)

def reduce_parse_CLA(argv):
  trainData_Path = ''
  testData_Path = ''
  trainOutput_Path = 'output_data'
  testOutput_Path = 'output_queries'
  for index, argument in enumerate(argv):
    if argument == '-d': trainData_Path = argv[index+1]
    if argument == '-q': testData_Path = argv[index+1]
    if argument == '-od': trainOutput_Path = argv[index+1]
    if argument == '-oq': testOutput_Path = argv[index+1]
  return trainData_Path,testData_Path,trainOutput_Path,testOutput_Path

def AE_read_Hyperparameters():
  print("Now provide the following hyperparameters:")
  conv_layers = int(input("Number of convolutional layers: "))
  kernel_size = int(input("Kernel size: "))

  conv_filters = []
  for i in range(conv_layers):
      filters = int(input("Convolutional layer %d, number of filters: " % i))
      conv_filters.append(filters)

  epochs = int(input("Number of epochs: "))
  batch_size = int(input("Batch size: "))

  return conv_layers, kernel_size, conv_filters, epochs, batch_size

def CL_read_Hyperparameters():
  print("Now provide the following hyperparameters:")
  fc_nodes = int(input("Number of nodes for the fully connected layer: "))
  epochs = int(input("Number of epochs: "))
  batch_size = int(input("Batch size: "))
  return fc_nodes,epochs,batch_size

def preprocess(data):
  data = data.reshape(-1, data.shape[1],data.shape[2], 1) #Reshape
  data = data / np.max(data) #Rescale
  return data
