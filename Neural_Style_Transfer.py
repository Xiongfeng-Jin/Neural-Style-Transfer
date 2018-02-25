import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import numpy as np
import tensorflow as tf
import cv2 as cv

class NeuralStyleTransfer:
	def __init__(self,contentImagePath,styleImagePath,modelPath):
		self.contentImage = cv.imread(contentImagePath)
		self.styleImage = cv.imread(styleImagePath)
		self.outputWidth = 0
		self.outputHeight = 0
		self.channels = 3
		self.noiseRatio = 0.6
		self.means = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3)) 
		if self.contentImage is None or self.styleImage is None:
			print("invalid image path")
			return
		self.__preprocessImages__()
		self.model = self.load_vgg_model(modelPath)
		self.STYLE_LAYERS = [
			('conv1_2', 4/42.5),
			('conv2_1', 4/42.5),
			('conv2_2', 4/42.5),
			('conv3_1', 3.5/42.5),
			('conv3_2', 3/42.5),
			('conv3_3', 2/42.5),
			('conv3_4', 1/42.5),
			('conv4_3', 2/42.5),
			('conv4_4', 3/42.5),
			('conv5_1', 4/42.5),
			('conv5_2', 4/42.5),
			('conv5_3', 4/42.5),
			('conv5_4', 4/42.5)]
		self.alpha = tf.placeholder(tf.float32)
		self.beta = 6
		self.learningRate = 2.0
		self.sess = tf.InteractiveSession()
		self.sess.run(self.model['input'].assign(self.contentImage))
		out = self.model['conv4_1']

		# Set a_C to be the hidden layer activation from the layer we have selected
		a_C = self.sess.run(out)

		# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
		# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
		# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
		a_G = out

		# Compute the content cost
		self.contentCost = self.compute_content_cost(a_C, a_G)

		# Assign the input of the model to be the "style" image 
		self.sess.run(self.model['input'].assign(self.styleImage))

		# Compute the style cost
		self.styleCost = self.compute_style_cost()

		self.totalCost = self.total_cost(self.contentCost, self.styleCost,self.alpha, self.beta)

		self.optimizer = tf.train.AdamOptimizer(self.learningRate)
		self.train_step = self.optimizer.minimize(self.totalCost)
		
		
	def __preprocessImages__(self):
		self.outputHeight = min(self.contentImage.shape[0], self.styleImage.shape[0])
		self.outputWidth = min(self.contentImage.shape[1], self.styleImage.shape[1])
		self.contentImage = cv.resize(self.contentImage,(self.outputWidth,self.outputHeight))
		self.styleImage = cv.resize(self.styleImage,(self.outputWidth,self.outputHeight))
		self.contentImage = self.reshape_and_normalize_image(self.contentImage)
		self.styleImage = self.reshape_and_normalize_image(self.styleImage)
		self.generatedImage = self.generate_noise_image(self.contentImage)
		
		
	def _weights(self,layer, expected_layer_name):
			"""
			Return the weights and bias from the VGG model for a given layer.
			"""
			wb = self.vgg_layers[0][layer][0][0][2]
			W = wb[0][0]
			b = wb[0][1]
			layer_name = self.vgg_layers[0][layer][0][0][0][0]
			assert layer_name == expected_layer_name
			return W, b

			return W, b

	def _relu(self,conv2d_layer):
			"""
			Return the RELU function wrapped over a TensorFlow layer. Expects a
			Conv2d layer input.
			"""
			return tf.nn.relu(conv2d_layer)

	def _conv2d(self,prev_layer, layer, layer_name):
			"""
			Return the Conv2D layer using the weights, biases from the VGG
			model at 'layer'.
			"""
			W, b = self._weights(layer, layer_name)
			W = tf.constant(W)
			b = tf.constant(np.reshape(b, (b.size)))
			return tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b

	def _conv2d_relu(self,prev_layer, layer, layer_name):
			"""
			Return the Conv2D + RELU layer using the weights, biases from the VGG
			model at 'layer'.
			"""
			return self._relu(self._conv2d(prev_layer, layer, layer_name))

	def _avgpool(self,prev_layer):
			"""
			Return the AveragePooling layer.
			"""
			return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
			
	def generate_noise_image(self,content_image):
		"""
		Generates a noisy image by adding random noise to the content_image
		"""
		
		# Generate a random noise_image
		noise_image = np.random.uniform(-20, 20, (1, self.outputHeight, self.outputWidth, self.channels)).astype('float32')
		
		# Set the input_image to be a weighted average of the content_image and a noise_image
		input_image = noise_image * self.noiseRatio + content_image * (1 - self.noiseRatio)
		
		return input_image


	def reshape_and_normalize_image(self,image):
		"""
		Reshape and normalize the input image (content or style)
		"""
		
		# Reshape image to mach expected input of VGG16
		image = np.reshape(image, ((1,) + image.shape))
		
		# Substract the mean to match the expected input of VGG16
		image = image - self.means
		
		return image


	def save_image(self,filename, image):
		
		# Un-normalize the image so that it looks good
		image = image + self.means
		
		# Clip and Save the image
		if not os.path.exists("output/"):
			os.makedirs("output/")
		image = np.clip(image[0], 0, 255).astype('uint8')
		cv.imwrite("output/"+filename,image)
			
	def load_vgg_model(self,path):
		"""
		Returns a model for the purpose of 'painting' the picture.
		Takes only the convolution layer weights and wrap using the TensorFlow
		Conv2d, Relu and AveragePooling layer. VGG actually uses maxpool but
		the paper indicates that using AveragePooling yields better results.
		The last few fully connected layers are not used.
		Here is the detailed configuration of the VGG model:
			0 is conv1_1 (3, 3, 3, 64)
			1 is relu
			2 is conv1_2 (3, 3, 64, 64)
			3 is relu    
			4 is maxpool
			5 is conv2_1 (3, 3, 64, 128)
			6 is relu
			7 is conv2_2 (3, 3, 128, 128)
			8 is relu
			9 is maxpool
			10 is conv3_1 (3, 3, 128, 256)
			11 is relu
			12 is conv3_2 (3, 3, 256, 256)
			13 is relu
			14 is conv3_3 (3, 3, 256, 256)
			15 is relu
			16 is conv3_4 (3, 3, 256, 256)
			17 is relu
			18 is maxpool
			19 is conv4_1 (3, 3, 256, 512)
			20 is relu
			21 is conv4_2 (3, 3, 512, 512)
			22 is relu
			23 is conv4_3 (3, 3, 512, 512)
			24 is relu
			25 is conv4_4 (3, 3, 512, 512)
			26 is relu
			27 is maxpool
			28 is conv5_1 (3, 3, 512, 512)
			29 is relu
			30 is conv5_2 (3, 3, 512, 512)
			31 is relu
			32 is conv5_3 (3, 3, 512, 512)
			33 is relu
			34 is conv5_4 (3, 3, 512, 512)
			35 is relu
			36 is maxpool
			37 is fullyconnected (7, 7, 512, 4096)
			38 is relu
			39 is fullyconnected (1, 1, 4096, 4096)
			40 is relu
			41 is fullyconnected (1, 1, 4096, 1000)
			42 is softmax
		"""
		
		vgg = scipy.io.loadmat(path)

		self.vgg_layers = vgg['layers']
		
		# Constructs the graph model.
		graph = {}
		graph['input']   = tf.Variable(np.zeros((1, self.outputHeight, self.outputWidth,self.channels)), dtype = 'float32')
		graph['conv1_1']  = self._conv2d_relu(graph['input'], 0, 'conv1_1')
		graph['conv1_2']  = self._conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
		graph['avgpool1'] = self._avgpool(graph['conv1_2'])
		graph['conv2_1']  = self._conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
		graph['conv2_2']  = self._conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
		graph['avgpool2'] = self._avgpool(graph['conv2_2'])
		graph['conv3_1']  = self._conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
		graph['conv3_2']  = self._conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
		graph['conv3_3']  = self._conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
		graph['conv3_4']  = self._conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
		graph['avgpool3'] = self._avgpool(graph['conv3_4'])
		graph['conv4_1']  = self._conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
		graph['conv4_2']  = self._conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
		graph['conv4_3']  = self._conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
		graph['conv4_4']  = self._conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
		graph['avgpool4'] = self._avgpool(graph['conv4_4'])
		graph['conv5_1']  = self._conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
		graph['conv5_2']  = self._conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
		graph['conv5_3']  = self._conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
		graph['conv5_4']  = self._conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
		graph['avgpool5'] = self._avgpool(graph['conv5_4'])
		
		return graph
		
	def compute_content_cost(self,a_C, a_G):
		"""
		Computes the content cost
		
		Arguments:
		a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
		a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
		
		Returns: 
		J_content -- scalar that you compute using equation 1 above.
		"""
		
		m, n_H, n_W, n_C = a_G.get_shape().as_list()
		
		a_C_unrolled = tf.reshape(tf.transpose(a_C),[n_C,n_W*n_H])
		a_G_unrolled = tf.reshape(tf.transpose(a_G),[n_C,n_W*n_H])
		
		J_content = 1/(4*n_C*n_W*n_H)*tf.reduce_sum(tf.subtract(a_C_unrolled,a_G_unrolled)**2)
		
		return J_content

	def gram_matrix(self,A):
		"""
		Argument:
		A -- matrix of shape (n_C, n_H*n_W)
		
		Returns:
		GA -- Gram matrix of A, of shape (n_C, n_C)
		"""
		GA = tf.matmul(A,tf.transpose(A))		
		return GA
		
		
	def compute_layer_style_cost(self,a_S, a_G):
		"""
		Arguments:
		a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
		a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
		
		Returns: 
		J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
		"""
		m, n_H, n_W, n_C = a_G.get_shape().as_list()
		
		a_S = tf.reshape(tf.transpose(a_S),[n_C,n_W*n_H])
		a_G = tf.reshape(tf.transpose(a_G),[n_C,n_W*n_H])

		GS = self.gram_matrix(a_S)
		GG = self.gram_matrix(a_G)

		J_style_layer = 1/(4*(n_C**2)*((n_W*n_H)**2))*tf.reduce_sum((GS-GG)**2)
		
		return J_style_layer	
		
	def compute_style_cost(self):
		"""
		Computes the overall style cost from several chosen layers
		
		Arguments:
		model -- our tensorflow model
		STYLE_LAYERS -- A python list containing:
							- the names of the layers we would like to extract style from
							- a coefficient for each of them
		
		Returns: 
		J_style -- tensor representing a scalar value, style cost defined above by equation (2)
		"""
		
		# initialize the overall style cost
		J_style = 0

		for layer_name, coeff in self.STYLE_LAYERS:

			# Select the output tensor of the currently selected layer
			out = self.model[layer_name]

			# Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
			a_S = self.sess.run(out)

			# Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
			# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
			# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
			a_G = out
			
			# Compute style_cost for the current layer
			J_style_layer = self.compute_layer_style_cost(a_S, a_G)

			# Add coeff * J_style_layer of this layer to overall style cost
			J_style += coeff * J_style_layer

		return J_style
		
		
	def total_cost(self,J_content, J_style, alpha = 10, beta = 40):
		"""
		Computes the total cost function
		
		Arguments:
		J_content -- content cost coded above
		J_style -- style cost coded above
		alpha -- hyperparameter weighting the importance of the content cost
		beta -- hyperparameter weighting the importance of the style cost
		
		Returns:
		J -- total cost as defined by the formula above.
		"""
		J = alpha*J_content+beta*J_style		
		return J

	def run(self, num_iterations = 6000):
		
		# Initialize global variables (you need to run the session on the initializer)
		self.sess.run(tf.global_variables_initializer())
		
		# Run the noisy input image (initial generated image) through the model. Use assign().
		self.sess.run(self.model['input'].assign(self.generatedImage))
		
		for i in range(num_iterations):
		
			# Run the session on the train_step to minimize the total cost

			Jc, Js = self.sess.run([ self.contentCost, self.styleCost])
			alpha = (Js / Jc)*3
			self.sess.run(self.train_step,feed_dict={self.alpha:alpha})
			
			# Compute the generated image by running the session on the current model['input']
			self.generatedImage = self.sess.run(self.model['input'])

			# Print every 20 iteration.
			if i%20 == 0:
				Jt, Jc, Js = self.sess.run([self.totalCost, self.contentCost, self.styleCost],feed_dict={self.alpha:alpha})
				print("Iteration " + str(i) + " :")
				print("alpha: %.4g" % alpha)
				print("total cost = %.4g" % Jt)
				print("content cost = %.4g" % Jc)
				print("style cost = %.4g" % Js)
				# self.save_image(str(i)+".jpg", self.generatedImage)
						
		# save last generated image
		self.save_image('generated_image.jpg', self.generatedImage)
			
			
nst = NeuralStyleTransfer("content.jpg", "style.jpg", "D:\Downloads/imagenet-vgg-verydeep-19.mat")
nst.run()