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
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


# %%
def classification(encoder,fc_nodes):
    # Flatten
    flat = Flatten()(encoder)
    # Fully connected
    dense = Dense(fc_nodes, activation='relu')(flat)
    # Dropout
    drop = Dropout(0.7)(dense)
    # Output
    classified = Dense(10, activation='softmax')(drop)
    return classified


# %%
if __name__ == "__main__":

  # Parse the command line arguments
  trainData_Path, testData_Path, trainLabels_Path, testLabels_Path, model_Path = CL_parse_CLA(sys.argv)
  # If any of the path arguments was not given (forgotten or running as a jupyter notebook), then ask for them
  if not trainData_Path: trainData_Path = input('Please provide the path of the training data:') # trainData_Path = "train-images-idx3-ubyte.gz"
  if not testData_Path: testData_Path = input('Please provide the path of the test data:') # testData_Path = "t10k-images-idx3-ubyte.gz"
  if not trainLabels_Path: trainLabels_Path = input('Please provide the path of the training labels:') # trainLabels_Path = "train-labels-idx1-ubyte.gz"
  if not testLabels_Path: testLabels_Path = input('Please provide the path of the test labels:') # testLabels_Path = "t10k-labels-idx1-ubyte.gz"
  if not model_Path: model_Path = input('Please provide the path of the autoencoder model:') # model_Path = "autoencoder_best.h5"
  # Read the all the datasets
  trainData = read_Data(trainData_Path)
  testData = read_Data(testData_Path)
  trainLabels = read_Labels(trainLabels_Path)
  testLabels = read_Labels(testLabels_Path)
  # Load the autoencoder model and keep its encoding part
  autoencoder = load_model(model_Path)
  autoencoder_length = len(autoencoder.layers)
  encoder_length = int(autoencoder_length/2) -1
  encoder = autoencoder.layers[encoder_length].output
  # Preprocess the data
  trainData = preprocess(trainData)
  testData = preprocess(testData)

  history = [] # a list of the trained models' loss history, the corresponding hyperparameters of each model and the models themselves
  choice = 1
  while choice != 0 :
    if choice == 1:
      # Read the hyperparameters
      hyperparameters = CL_read_Hyperparameters()
      fc_nodes, epochs, batch_size = hyperparameters
      # Create the full model
      full_model = Model(autoencoder.input,classification(encoder,fc_nodes))
      # Set the weights of the encoding part of the full model to be the weights of the encoding part of the autoencoder model we loaded
      for layer_full,layer_encoder in zip(full_model.layers[:encoder_length],autoencoder.layers[:encoder_length]):
          layer_full.set_weights(layer_encoder.get_weights())
      # Set the layers of the encoding part of the full model to non-trainable
      for layer in full_model.layers[0:encoder_length]:
          layer.trainable = False
      full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=RMSprop(lr=0.001),metrics=['accuracy'])
      # Train the non-encoding part of the full model
      classify_train = full_model.fit(
          trainData, 
          to_categorical(trainLabels), 
          batch_size=batch_size,
          epochs=10,
          verbose=1,
          validation_split=0.2,
          shuffle=True
      )
      # Set back to trainable the layers of the encoding part of the full model 
      for layer in full_model.layers[0:encoder_length]:
          layer.trainable = True
      full_model.compile(loss=keras.losses.categorical_crossentropy, optimizer=RMSprop(lr=0.001), metrics=['accuracy'])
      # Train the whole full model
      classify_train = full_model.fit(
          trainData, 
          to_categorical(trainLabels), 
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=0.2,
          shuffle=True
      )
      history.append((classify_train,hyperparameters,full_model))
    elif choice == 2:
      # Plot each model's training and validation loss and accuracy
      for index,triple in enumerate(history):
        classify_train, hyperparams, model = triple
        fc_nodes, epochs, batch_size = hyperparams
        print(f'Model {index}: Nodes of Fully Connected Layer = {fc_nodes}, Epochs = {epochs}, Batch Size = {batch_size}')
        plt.subplot(1,2,1)
        plt.plot(classify_train.history['loss'])
        plt.plot(classify_train.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.tight_layout()
        plt.subplot(1,2,2)
        plt.plot(classify_train.history['accuracy'])
        plt.plot(classify_train.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.tight_layout()
        plt.show()        
    elif choice == 3:
      # Classify the test data according to one of the models and print some statistics
      model_num = int(input(f"Which model from {0} to {len(history)-1} would you like to use?:"))
      test_eval = history[model_num][2].evaluate(testData, to_categorical(testLabels), verbose=0)
      print('\nTest loss:', test_eval[0])
      print('Test accuracy:', test_eval[1])
      # Print the number of correctly and incorrect classified images, and a sample of each
      predicted_classes = history[model_num][2].predict(testData)
      predicted_classes = np.argmax(np.round(predicted_classes),axis=1)
      correct = np.where(predicted_classes==testLabels)[0]
      incorrect = np.where(predicted_classes!=testLabels)[0]
      print(f"\nFound -{len(correct)}- CORRECT labels. Some of them are:")
      for i, correct in enumerate(correct[:9]):
          plt.subplot(3,3,i+1)
          plt.imshow(testData[correct].reshape(28,28), cmap='gray', interpolation='none')
          plt.title(f"Predicted {predicted_classes[correct]}, Actual {testLabels[correct]}")
          plt.tight_layout()
      plt.show()
      print(f"\nFound {len(incorrect)} INCORRECT labels. Some of them are:")
      for i, incorrect in enumerate(incorrect[:9]):
          plt.subplot(3,3,i+1)
          plt.imshow(testData[incorrect].reshape(28,28), cmap='gray', interpolation='none')
          plt.title(f"Predicted {predicted_classes[incorrect]}, Actual {testLabels[incorrect]}")
          plt.tight_layout()
      plt.show()
      # Print the classification report (f-score etc.)
      target_names = [f"Class {i}" for i in range(10)]
      print(classification_report(testLabels, predicted_classes, target_names=target_names))
    choice = int(input("What would you like to do next?\n 0) Exit\n 1) Repeat the experiment with different parameters\n 2) Plot the metrics for each experiment\n 3) Classify the images in the test set\n"))


# %%
# This deletes the cloned repository from the current colab session. You don't have to run it, since when the session ends, all files are deleted
if 'google.colab' in sys.modules:
  get_ipython().run_line_magic('cd', '..')
  get_ipython().system('rm -rf Project-2')


