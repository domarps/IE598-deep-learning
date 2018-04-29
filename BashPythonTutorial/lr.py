import numpy as np  
from chainer import datasets

# load datasets
train, test = datasets.get_mnist()


x_train = []
y_train = []

for x,y in train:
	x_train.append(x)
	y_train.append(y)

x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
#np.unique(ytrain)

x_test = []
y_test = []

for x,y in test:
	x_test.append(x)
	y_test.append(y)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)

training_size  = x_train.shape[0]
input_size = x_train.shape[1]
output_size = np.unique(y_train).shape[0]


#z = wX + b
#W : (10 * 784)
#b : 10
#a = g(Z)
#W = W + eta * W_{grad}

W = np.random.randn(output_size, input_size)/np.sqrt(input_size)
b = np.random.randn(output_size)/np.sqrt(output_size)


W_grad = np.zeros((output_size, input_size))
b_grad = np.zeros((output_size))

learning_rate = 0.001


def softmax(z):
	a = np.exp(z)
	a = a/a.sum()
	return a

def forward_pass(W,b,x):
	z = np.dot(W,x) + b
	a = softmax(z)
	return a


def calc_gradient(y,a,x):
	for j in xrange(output_size):
		delta = 1.0 * (y == j) - a[j] 
		W_grad[j,:] = delta*x #element-wise multiplication
		b_grad[j] = delta * 1.0
	return (W_grad, b_grad)

def update_weights(W, b, W_grad, b_grad, learning_rate):
	W = W + learning_rate * W_grad
	b = b + learning_rate * b_grad
	return (W,b)

for epoch in xrange(10):
	random_indices = np.random.permutation(training_size)
	correct = 0.0
	for index in random_indices:
		x = x_train[index]
		y = y_train[index]
		a1 = forward_pass(W, b1, x)
		(W_grad, b_grad) = calc_gradient(y, a, x)
		(W, b) = update_weights(W, b, W_grad, b_grad, learning_rate)
		y_hat = a.argmax()
		correct += 1.0 * (y==y_hat)
	train_accuracy = correct/training_size

y_hat = np.zeros(y_test.shape)
for i in xrange(y_test.shape[0]):
	x = x_test[i]
	y = y_test[i]
	a = forward_pass(W, b, x)
	y_hat[i] = a.argmax()

test_accuracy = 1.0 * np.sum(y_hat == y_test)/y_test.shape[0]
print "Train accuracy : ", str(train_accuracy), "Test accuracy : " , str(test_accuracy)


