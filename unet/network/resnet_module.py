from .basic_ops import *
import paddle

"""This script defines non-attention same-, up-, down- modules.
Note that pre-activation is used for residual-like blocks.
Note that the residual block could be used for downsampling.
"""


def res_block(inputs, output_filters, training, dimension,name=None):
	"""Standard residual block with pre-activation.
	Args:
		inputs: a Tensor with shape [batch, (d,) h, w, channels]
		output_filters: an integer
		training: a boolean for batch normalization and dropout
		dimension: a string, dimension of inputs/outputs -- 2D, 3D
		name: a string
		
	Returns:
		A Tensor of shape [batch, (_d,) _h, _w, output_filters]
	"""
	if dimension == '2D':
		convolution = convolution_2D
	elif dimension == '3D':
		convolution = convolution_3D
	else:
		raise ValueError("Dimension (%s) must be 2D or 3D." % (dimension))

	if inputs.shape[-1] == output_filters:
		shortcut = inputs
		inputs = batch_norm(inputs, training)
		inputs = relu(inputs, 'relu_1')
	else:
		inputs = batch_norm(inputs, training)
		inputs = relu(inputs, 'relu_1')
		shortcut = convolution(inputs.shape[-1], output_filters, 1, 1, False)(inputs)
	inputs = convolution(inputs.shape[-1], output_filters, 3, 1, False)(inputs)
	inputs = batch_norm(inputs, training)
	inputs = relu(inputs, 'relu_2')
	inputs = convolution(inputs.shape[-1], output_filters, 3, 1, False)(inputs)
	return paddle.add(shortcut, inputs)


def down_res_block(inputs, output_filters, training, dimension, name=None):
	"""Standard residual block with pre-activation for downsampling."""
	if dimension == '2D':
		convolution = convolution_2D
	elif dimension == '3D':
		convolution = convolution_3D
	else:
		raise ValueError("Dimension (%s) must be 2D or 3D." % (dimension))


		# The projection_shortcut should come after the first batch norm and ReLU.
	inputs = batch_norm(inputs, training)
	inputs = relu(inputs, 'relu_1')
	shortcut = convolution(inputs.shape[-1], output_filters, 1, 2, False)(inputs)
	inputs = convolution(inputs.shape[-1], output_filters, 3, 2, False)(inputs)
	inputs = batch_norm(inputs, training)
	inputs = relu(inputs, 'relu_2')
	inputs = convolution(inputs.shape[-1], output_filters, 3, 1, False)(inputs)
	return paddle.add(shortcut, inputs)

def down_convolution(inputs, output_filters, training, dimension,name=None):
	"""Use a single stride 2 convolution for downsampling."""
	if dimension == '2D':
		convolution = convolution_2D
	elif dimension == '3D':
		convolution = convolution_3D
	else:
		raise ValueError("Dimension (%s) must be 2D or 3D." % (dimension))


	inputs = convolution(inputs.shape[-1], output_filters, 3, 2, False)(inputs)
	return inputs

def up_transposed_convolution(inputs, output_filters, training, dimension,name=None):
	"""Use a single stride 2 transposed convolution for upsampling."""
	if dimension == '2D':
		transposed_convolution = transposed_convolution_2D
	elif dimension == '3D':
		transposed_convolution = transposed_convolution_3D
	else:
		raise ValueError("Dimension (%s) must be 2D or 3D." % (dimension))


	inputs = transposed_convolution(inputs.shape[-1], output_filters, 3, 2, False)(inputs)
	return inputs
