# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import sys, os
# If this is run on colab, it clones the git repository, so that you don't have to upload the datasets on your google drive
if 'google.colab' in sys.modules:
  get_ipython().system('git clone https://github.com/myioannis/Project-2.git')
  # Change to the directory of the cloned repository
  get_ipython().run_line_magic('cd', 'Project-2')
  sys.path.append(os.getcwd())


# %%
from utilities import read_Data, read_Labels, AE_parse_CLA, CL_parse_CLA, AE_read_Hyperparameters, CL_read_Hyperparameters, preprocess #from our utilities.py
import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
import matplotlib.pyplot as plt


# %%
# The encoding part of the autoencoder
def encoder(input_img, hyperparameters):
    # Read the hyperparameters
    conv_layers, kernel_size, conv_filters = hyperparameters[:3]
    # First convolutional layer (at least one)
    conv = Conv2D(conv_filters[0], (kernel_size, kernel_size), activation='relu', padding='same')(input_img)
    conv = BatchNormalization()(conv)
    if conv.shape[1] % 2 == 0:
        # Downsample image by half if possible
        conv = MaxPooling2D((2, 2))(conv)
        # Dropout to avoid overfitting
        conv = Dropout(0.2)(conv)

    # Remaining convolutional layers
    for i in range(1, conv_layers):
        conv = Conv2D(conv_filters[i], (kernel_size, kernel_size), activation='relu', padding='same')(conv)
        conv = BatchNormalization()(conv)
        if conv.shape[1] % 2 == 0:
            conv = MaxPooling2D((2, 2))(conv)
            conv = Dropout(0.2)(conv)
    return conv

# The decoding part of the autoencoder
def decoder(conv, hyperparameters):
    conv_layers, kernel_size, conv_filters = hyperparameters[:3]
    for i in reversed(range(conv_layers)):
        conv = Conv2D(conv_filters[i], (kernel_size, kernel_size), activation='relu', padding='same')(conv)
        conv = BatchNormalization()(conv)
        if i <= 1:
            # For 28x28 images, downsampling happens only 2 times (28x28 -> 14x14 -> 7x7)
            conv = UpSampling2D((2, 2))(conv)
            conv = Dropout(0.2)(conv)
    decoded = Conv2D(1, (kernel_size, kernel_size), activation='sigmoid', padding='same')(conv)
    return decoded


# %%
if __name__ == "__main__":

  # Parse the command line arguments
  trainData_Path = AE_parse_CLA(sys.argv)
  # If any of the path arguments was not given (forgotten or running as a jupyter notebook), then ask for them
  if not trainData_Path: trainData_Path = input('Please provide the path of the training data: ')  #trainData_Path = "train-images-idx3-ubyte.gz"
  # Read the datasets
  trainData = read_Data(trainData_Path)
  # Preprocess the data
  trainData = preprocess(trainData)

  history = [] # a list of the trained models' loss history and the corresponding hyperparameters of each model
  choice = 1
  while choice != 0 :
    if choice == 1:
      # Ask the user for the hyperparameters
      hyperparameters = AE_read_Hyperparameters()
      epochs, batch_size = hyperparameters[3:]
      # Create the model
      input_img = Input(shape = (trainData.shape[1], trainData.shape[2], 1))
      autoencoder = Model(input_img, decoder(encoder(input_img,hyperparameters),hyperparameters))
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
      history.append((autoencoder_train,hyperparameters))
    elif choice == 2:
      # Plot the loss and validation loss of each model
      for model,hyperparams in history:
        conv_layers, kernel_size, conv_filters, epochs, batch_size = hyperparams
        print(f'Model: Convolutional Layers = {conv_layers}, Kernel Size = {kernel_size}, Convolutional Filters = {conv_filters}, Epochs = {epochs}, Batch Size = {batch_size}')
        plt.plot(model.history['loss'])
        plt.plot(model.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()        
    elif choice == 3:
      # Save the (whole) model
      model_Path = input("Where would you like the model to be saved? Provide relative path: ")
      autoencoder.save(model_Path)
    choice = int(input("What would you like to do next?\n 0) Exit\n 1) Repeat the experiment with different parameters\n 2) Plot the loss for each experiment\n 3) Save the most recent -whole- model\n"))
    


# %%
# This deletes the cloned repository from the current colab session. You don't have to run it, since when the session ends, all files are deleted
if 'google.colab' in sys.modules:
  get_ipython().run_line_magic('cd', '..')
  get_ipython().system('rm -rf Project-2')


