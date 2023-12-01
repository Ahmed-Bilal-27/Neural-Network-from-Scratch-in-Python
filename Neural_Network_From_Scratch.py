"""
This code implements neural network from scratch without any neural network library
(e.g. TensorFlow, PyTorch, Keras etc) using 01 input, 01 hidden and 01 output layer.
The dataset used is:
    Dataset Name: Digit Recognizer
    Dataset URL: https://www.kaggle.com/competitions/digit-recognizer/data?select=sample_submission.csv#
"""
#----------------Imports-----------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#----------------Imports-----------------
# Reading data in a pandas dataframe
data = pd.read_csv('D:\\Programs\\Python\\Neural_Network_From_Scratch\\Dataset\\train.csv')
# Converting data from pandas dataframe into numpy array to perform Linear Algebra functions
data = np.array(data)
# Printing data shape
print(data.shape)
# Storing data array rows and columns in m and n variable
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into test and training sets
# Splitting into test and training data
data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255
# Printing testing data labels
print(Y_test)
data_train = data[1000:m].T
# Printing testing data labels
Y_train = data_train[0]
print(Y_train)
X_train = data_train[1:n]
X_train = X_train / 255
# Defining method to initialize weights and biases matrices
def init_params():
    # np.random.rand() - Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
    # Link: https://numpy.org/doc/stable/reference/random/generated/numpy.random.rand.html
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2
# Defining ReLU function which is used in layer 01 (first hidden layer) where if x >= 0 then y = x else y = 0
def ReLU(Z):
    # You can either implement it using if else or using this maximum function
    # np.maximum() - Compare two arrays and return a new array containing the element-wise maxima.
    # Link: https://numpy.org/doc/stable/reference/generated/numpy.maximum.html
    return np.maximum(Z, 0)
# Defining softmax function used in layer 02 (output layer)
def softmax(Z):
    # np.exp() - Calculate the exponential of all elements in the input array.
    # Link: https://numpy.org/doc/stable/reference/generated/numpy.exp.html
    A = np.exp(Z) / sum(np.exp(Z))
    return A
# Defining forward propagation function
def forward_propagation(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 # Calculating dot product of weight and input and adding bias for layer 01 (first hidden layer)
    A1 = ReLU(Z1) # Applying ReLU activation function to the layer 01 (first hidden layer)
    Z2 = W2.dot(A1) + b2 # Calculating dot product of weight and layer 01 (first hidden layer) output and adding bias for layer 02 (output layer)
    A2 = softmax(Z2) # Applying softmax activation function to the layer 02 (output layer)
    return Z1, A1, Z2, A2
# Calculating derivative of ReLU activation function of layer 01 (first hidden layer) for backward propagation
def ReLU_deriv(Z):
    '''
    The derivative of ReLU is:
        f(x)={0 if x < 0
             {1 if x > 0
    And undefined at x = 0. The reason for it being undefined at x=0 is that its left- and right derivative are not equal.
    '''
    return Z > 0
# Doing one-hot encoding for Y labels to calculate absolute error(loss function)
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
# Defining backward propagation function
def backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2
# In update parameter we are updating parameters based upon the result from backward propagation
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
# Defining method to make predictions on data
def get_predictions(A2):
    return np.argmax(A2, 0)
# Defining function to calculate accuracy
def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size
# Defining Gradient Descent method for optimization
def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2
# Training the neural network
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.10, 500)
# Making predictions on test data
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label) 
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)
test_prediction(8, W1, b1, W2, b2)
dev_predictions = make_predictions(X_test, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_test)