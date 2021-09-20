import sys, os
from utilities import read_Data, read_Labels, reduce_parse_CLA, AE_read_Hyperparameters, preprocess #from our utilities.py
import numpy as np
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Reshape
from tensorflow.keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from math import ceil
import struct
import matplotlib.pyplot as plt

strides = (2,2)

def conv_output_shape(in_height,in_width,strides):
  """ Returns the expected output height and width of the --conv2D-- layer after --downsampling-- with strides """
  out_height = ceil(float(in_height) / strides[0])
  out_width  = ceil(float(in_width) / strides[1])
  return (out_height,out_width)

def trans_output_shape(in_height,in_width,strides):
  """ Returns the expected output height and width of the --conv2DTranspose-- layer after --upsampling-- with strides """
  out_height = in_height * strides[0]
  out_width  = in_width * strides[1]
  return (out_height,out_width)

def find_padding(in_height,in_width):
  """ Finds the right padding to be forced on the conv2DTranspose to achieve upsampling symmetrical to the conv2D downsampling """
  expected_conv_height, expected_conv_width = conv_output_shape(in_height,in_width,strides)
  expected_trans_height, expected_trans_width = trans_output_shape(expected_conv_height,expected_conv_width,strides)
  return (in_height-expected_trans_height+1,in_width-expected_trans_width+1)

def find_encoding_output(model,num_latent_dimensions):
  for layer_num,layer in enumerate(model.layers):
    if isinstance(layer,keras.layers.Dense) and layer.output_shape[1] == num_latent_dimensions:
      return layer.output

def encoder(conv, hyperparameters, num_latent_dimensions):
  """ The encoding part of the autoencoder """
  # Read the hyperparameters
  conv_layers, kernel_size, conv_filters = hyperparameters[:3]
  # Keep a list of the shape of each encoding layer --after downsampling-- in order to help the decoder do a --symmetrical upsampling--
  encoding_shapes = []
  # First convolutional layers
  for i in range(conv_layers-1):
      conv = BatchNormalization()(conv)
      out_height,out_width = conv_output_shape(conv.shape[1],conv.shape[2],strides)
      if out_height > 0 and out_width > 0:
        conv = Conv2D(conv_filters[i], (kernel_size, kernel_size),  activation='relu', strides=strides, padding='same')(conv)
        encoding_shapes.append((out_height,out_width))
        conv = Dropout(0.3)(conv)
      else:
        # Do not downsample (no strides)
        conv = Conv2D(conv_filters[i], (kernel_size, kernel_size), activation='relu', padding='same')(conv)

  # Last convolutional layer
  out_height,out_width = conv_output_shape(conv.shape[1],conv.shape[2],strides)
  if out_height > 0 and out_width > 0:
    conv = Conv2D(conv_filters[conv_layers-1], (kernel_size, kernel_size),  activation='relu', strides=strides, padding='same')(conv)
    encoding_shapes.append((out_height,out_width))
  else:
    # Do not downsample (no strides)
    conv = Conv2D(conv_filters[conv_layers-1], (kernel_size, kernel_size), activation='relu', padding='same')(conv)

  conv = Flatten()(conv)
  conv = Dense(num_latent_dimensions, activation='relu')(conv)
  return conv,tuple(encoding_shapes)



def decoder(encoder, hyperparameters):
  """ The decoding part of the autoencoder """
  conv_layers, kernel_size, conv_filters = hyperparameters[:3]
  conv, encoding_shapes = encoder

  in_height, in_width = encoding_shapes[-1]
  print(encoding_shapes)
  conv = Dense(in_height*in_width*conv_filters[-1], activation='relu')(conv)
  conv = Reshape((in_height,in_width,conv_filters[-1]))(conv)

  # First convolutional layers
  for i in reversed(range(conv_layers-1)):
      conv = BatchNormalization()(conv)
      if i <= len(encoding_shapes)-1:
        in_height, in_width = encoding_shapes[i]
        conv = Conv2DTranspose(conv_filters[i], (kernel_size, kernel_size), strides=strides, output_padding=find_padding(in_height,in_width), padding='same', activation='relu')(conv)
        conv = Dropout(0.3)(conv)
      else:
        # Do not upsample
        conv = Conv2DTranspose(conv_filters[i], (kernel_size, kernel_size), activation='relu')(conv)

  # Last convolutional layer
  decoded = Conv2DTranspose(1, (kernel_size, kernel_size), strides=strides, padding='same', activation='sigmoid')(conv)
  return decoded

def init_output_file(filename, numOfImages, numOfRows, numOfColumns):
  """ Initializes the output file with magicNumber, numOfImages, numOfRows, numOfColumns """
  with open(filename, 'wb') as file:
    file.write(struct.pack('>i', 42))
    file.write(struct.pack('>i', numOfImages))
    file.write(struct.pack('>i', numOfRows))
    file.write(struct.pack('>i', numOfColumns))

def write_to_output(filename, vector):
  """ Writes latent representation of an image in the output file (high-endian) """
  with open(filename, 'ab') as file:
    for pixel in vector:
      file.write(struct.pack('>H', pixel))

def latent_dimensions(encoder_model):
  """ Returns the number of latent dimensions of a --pretrained-- model """
  return encoder_model.layers[-1].output_shape[1]

if __name__ == "__main__":

  # Parse the command line arguments
  trainData_Path, testData_Path, trainOutput_Path, testOutput_Path = reduce_parse_CLA(sys.argv)
  # If any of the path arguments was not given (forgotten or running as a jupyter notebook), then ask for them
  if not trainData_Path: trainData_Path = input('Please provide the path of the training data: ')
  if not testData_Path: testData_Path = input('Please provide the path of the test data: ')

  # Read the datasets
  trainData = read_Data(trainData_Path) #train-images-idx3-ubyte.gz
  testData = read_Data(testData_Path) #t10k-images-idx3-ubyte.gz

  # Preprocess the data
  trainData = preprocess(trainData)
  testData = preprocess(testData)

  history = [] # a list of the trained models' loss history, the corresponding hyperparameters of each model and the models themselves
  choice = "start"
  num_latent_dimensions = 0
  while choice != 0 :
    if choice == "start":
      inner_choice = int(input("What would you like to do?\n 0) Exit\n 1) Upload a pretrained --encoder-- model and convert the images to their latent representation\n 2) Train your own model\n"))
      
      if inner_choice == 1:
        # Upload a pretrained --encoder-- model
        model_Path = input('Please provide the path of the --encoder-- model: ')
        encoder_model = load_model(model_Path)

        # Predict (convert into latent representations) on the data and query sets
        encoded_data = encoder_model.predict(trainData)
        encoded_queries = encoder_model.predict(testData)

        print("latent_dimensions: ",latent_dimensions(encoder_model))
        # Initialize the output files
        init_output_file(trainOutput_Path, len(trainData), 1, latent_dimensions(encoder_model))
        init_output_file(testOutput_Path, len(testData), 1, latent_dimensions(encoder_model))

        # Write the latent representations of the -data- into the output file
        encoded_max = np.amax(encoded_data)
        print(encoded_max)
        for i in range(len(trainData)):
          encoded_img = encoded_data[i]
          encoded_img *= 25500.0/encoded_max
          encoded_img = encoded_img.astype(int)
          write_to_output(trainOutput_Path, encoded_img)
        
        # Write the latent representations of the -queries- into the output file
        encoded_max = np.amax(encoded_queries)
        print(encoded_max)
        for i in range(len(testData)):
          encoded_img = encoded_queries[i]
          encoded_img *= 25500.0/encoded_max
          encoded_img = encoded_img.astype(int)
          write_to_output(testOutput_Path, encoded_img)
  
      elif inner_choice == 2:
        choice = 1
        continue

      else:
        break

    elif choice == 1:
      # Ask the user for the hyperparameters
      hyperparameters = AE_read_Hyperparameters()
      epochs, batch_size = hyperparameters[3:]
      num_latent_dimensions = int(input("How many dimensions would you like the latent representations of the images to be? \n"))
      
      # Create the model
      input_img = Input(shape = (trainData.shape[1], trainData.shape[2], 1))
      autoencoder = Model(input_img, decoder(encoder(input_img,hyperparameters,num_latent_dimensions),hyperparameters))
      print(autoencoder.summary())
      autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop(lr=0.001))
      # Train the model
      autoencoder_train = autoencoder.fit(
          trainData, 
          trainData, 
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2,
          shuffle=True
      )
      history.append((autoencoder_train,hyperparameters,autoencoder))

    elif choice == 2:
      # Plot the loss and validation loss of each model
      for index,triple in enumerate(history):
        autoencoder_train, hyperparams, model = triple
        conv_layers, kernel_size, conv_filters, epochs, batch_size = hyperparams
        print(f'Model {index}: Convolutional Layers = {conv_layers}, Kernel Size = {kernel_size}, Convolutional Filters = {conv_filters}, Epochs = {epochs}, Batch Size = {batch_size}')
        plt.plot(autoencoder_train.history['loss'])
        plt.plot(autoencoder_train.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()        

    elif choice == 3:
      # Save the --encoding-- part of one of the models
      model_num = int(input(f"Which model from {0} to {len(history)-1} would you like to save?:"))
      model = history[model_num][2]
      encoder_output = find_encoding_output(model,num_latent_dimensions)
      encoder_model = Model(model.input,encoder_output)
      model_Path = input("Where would you like the encoder to be saved? Provide relative path: ")
      encoder_model.save(model_Path)

    elif choice == 4:
      model_num = int(input(f"Which model from {0} to {len(history)-1} would you like to use?:"))
      model = history[model_num][2]
      encoder_output = find_encoding_output(model,num_latent_dimensions)
      encoder_model = Model(model.input,encoder_output)

      # Predict (convert into latent representations) on the data and query sets
      encoded_data = encoder_model.predict(trainData)
      encoded_queries = encoder_model.predict(testData)

      # Initialize the output files
      init_output_file(trainOutput_Path, len(trainData), 1, num_latent_dimensions)
      init_output_file(testOutput_Path, len(testData), 1, num_latent_dimensions)

      # Write the latent representations of the -data- into the output file
      encoded_max = np.amax(encoded_data)
      print(encoded_max)
      for i in range(len(trainData)):
        encoded_img = encoded_data[i]
        encoded_img *= 25500.0/encoded_max
        encoded_img = encoded_img.astype(int)
        write_to_output(trainOutput_Path, encoded_img)
      
      # Write the latent representations of the -queries- into the output file

      encoded_max = np.amax(encoded_queries)
      print(encoded_max)
      for i in range(len(testData)):
        encoded_img = encoded_queries[i]
        encoded_img *= 25500.0/encoded_max
        encoded_img = encoded_img.astype(int)
        write_to_output(testOutput_Path, encoded_img)

    choice = int(input("What would you like to do next?\n 0) Exit\n 1) Train with different parameters\n 2) Plot the loss for each model\n 3) Save the --encoding-- part of one of the models\n 4) Convert the images into their latent representations using one of the trained models\n"))
