
# coding: utf-8

# In[1]:

#package imports
import numpy as np  
from chainer import datasets


# In[2]:

# load datasets
train, test = datasets.get_mnist()


# In[3]:

x_train = []
y_train = []

for x,y in train:
	x_train.append(x)
	y_train.append(y)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)


# In[4]:

x_test = []
y_test = []

for x,y in test:
	x_test.append(x)
	y_test.append(y)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)


# In[5]:

#training_size = number of examples
#input_size = input dimensionality
#output_size = number of classes
training_size  = x_train.shape[0]
input_size = x_train.shape[1]
output_size = np.unique(y_train).shape[0]


# In[6]:

num_hidden_units = 200
num_epochs = 40
learning_rate = 0.001


# In[7]:

W1 = np.random.rand(num_hidden_units, input_size)/np.sqrt(input_size)
b1 = np.random.rand(num_hidden_units)/np.sqrt(num_hidden_units)

W2 = np.random.rand(output_size, num_hidden_units)/np.sqrt(num_hidden_units)
b2 = np.random.rand(output_size)/np.sqrt(output_size)

W1_grad = np.zeros((num_hidden_units, input_size))
b1_grad = np.zeros((num_hidden_units))
W2_grad = np.zeros((output_size, num_hidden_units))
b2_grad = np.zeros((output_size))


# In[8]:

def relu_grad(z):
    a = 1.0*(z>=0)
    return a

def forward_pass_1(W1,b1,x):
    z1 = np.dot(W1,x) + b1
    a = np.array(z1*(z1>=0))
    return a

def forward_pass_2(W2,b2,a2):
    z2 = np.dot(W2,a2) + b2
    g = np.exp(z2)
    g = g/g.sum()
    return g


def calc_gradient(y,a,g,x,W1,b1,W2,b2):
    
    z1 = W1.dot(x)+b1
    
    class_list = np.array(xrange(output_size))
    delta_3 = (y==class_list)-g
    delta_2 = delta_3.dot(W2)*relu_grad(z1)
    
    W2_grad = np.outer(delta_3,a)
    b2_grad = delta_3
    W1_grad = np.outer(delta_2,x)
    b1_grad = delta_2
    
    return W1_grad,b1_grad,W2_grad,b2_grad

def update_weights(W, b, W_grad, b_grad, learning_rate):
    W = W + learning_rate * W_grad
    b = b + learning_rate * b_grad
    return (W,b)


# In[9]:

for epoch in xrange(num_epochs):
    print "At epoch number", epoch
    random_indices = np.random.permutation(training_size)
    correct = 0.0
    for index in random_indices:
        x = x_train[index]
        y = y_train[index]
        
        a2 = forward_pass_1(W1, b1, x)
        g = forward_pass_2(W2, b2, a2)
        
        (W1_grad,b1_grad,W2_grad,b2_grad) = calc_gradient(y,a2,g,x,W1,b1,W2,b2)
        (W1, b1) = update_weights(W1, b1, W1_grad, b1_grad, learning_rate)
        (W2, b2) = update_weights(W2, b2, W2_grad, b2_grad, learning_rate)
        
        y_hat = g.argmax()
        
        correct += 1.0 * (y==y_hat)
        
    train_accuracy = correct/training_size

    y_hat = np.zeros(y_test.shape)
    for i in xrange(y_test.shape[0]):
        x=x_test[i]
        y=y_test[i]
        a2 = forward_pass_1(W1,b1,x)
        g = forward_pass_2(W2,b2,a2)
        y_hat[i] = g.argmax()
    
    test_accuracy = 1.0*np.sum(y_hat==y_test)/y_test.shape[0]
    
    print "train accuracy: ",train_accuracy, "test accuracy", test_accuracy



