import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy.random


# neural network config
learning_rate = 0.001
n_input_neurons = 9
n_hidden_neurons_layer_1 = 18
n_hidden_neurons_layer_2 = 18
n_output_neurons = 9

weights = {
	# syntaptic weight from input -> hidden layer 1: size (9,18)
	'h1': tf.Variable(tf.random_normal([n_input_neurons, n_hidden_neurons_layer_1])),
	# syntaptic weight from hidden layer 1 -> hidden layer 2: size (18,18)
	'h2': tf.Variable(tf.random_normal([n_hidden_neurons_layer_1, n_hidden_neurons_layer_2])),
	# syntaptic weight from hidden layer 2 -> output layer: size (18,9)
	'out': tf.Variable(tf.random_normal([n_hidden_neurons_layer_2, n_output_neurons]))
}

biases = {
	# bias from input -> hidden layer 1: size (18)
	'b1': tf.Variable(tf.random_normal([n_hidden_neurons_layer_1])),
	# bias from hidden layer 1 -> hidden layer 2: size (18)
	'b2': tf.Variable(tf.random_normal([n_hidden_neurons_layer_2])),
	# bias from hidden layer 2 -> output layer: size (9)
	'out': tf.Variable(tf.random_normal([n_output_neurons]))
}

def create_multi_layer_perceptron(x, weights, biases):
	# hidden layer 1
	layer_1 = tf.matmul(x, weights['h1']) + biases['b1']
	layer_1 = tf.nn.relu(layer_1)
	# hidden layer 2
	layer_2 = tf.matmul(layer_1, weights['h2']) + biases['b2']
	layer_2 = tf.nn.relu(layer_2)
	# output layer
	layer_out = tf.matmul(layer_2, weights['out']) + biases['out']
	return layer_out

def train_network():

	X = tf.placeholder('float', shape=(None, n_input_neurons))
	y = tf.placeholder('float', shape=(None, n_output_neurons))

	network = create_multi_layer_perceptron(X, weights, biases)

	# possibly switch cost function in the future
	loss = tf.nn.softmax_cross_entropy_with_logits(predictions, y)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	train = optimizer.minimize(loss)

# game setup
# generates a list of all game states, actions, and rewards
def generate_game_states():
	# rewards
	win = 10
	tie = 5
	loss = 0
	# randomly choose which side starts
	starting_side = random.choose([-1, 1])





