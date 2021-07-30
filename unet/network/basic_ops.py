import sys
sys.path.append('../..')
from network_configure import conf_basic_ops
from paddle import nn
import paddle

"""This script defines basic operaters.
"""

def convolution_2D(in_channels, filters, kernel_size, strides, use_bias):
	"""Performs 2D convolution without activation function.
	If followed by batch normalization, set use_bias=False.
	"""
	return nn.Conv2D(
				in_channels=in_channels,
				out_channels=filters,
				kernel_size=kernel_size,
				stride=strides,
				padding='SAME',
				bias_attr=use_bias,
			)

def convolution_3D(in_channels, filters, kernel_size, strides, use_bias):
	"""Performs 3D convolution without activation function.
	If followed by batch normalization, set use_bias=False.
	"""
	return nn.Conv3D(
				in_channels=in_channels,
				out_channels=filters,
				kernel_size=kernel_size,
				stride=strides,
				padding='SAME',
				bias_attr=use_bias,
				data_format="NDHWC"
			)

def transposed_convolution_2D(in_channels, filters, kernel_size, strides, use_bias):
	"""Performs 2D transposed convolution without activation function.
	If followed by batch normalization, set use_bias=False.
	"""
	return nn.Conv2DTranspose(
				in_channels=in_channels,
				out_channels=filters,
				kernel_size=kernel_size,
				stride=strides,
				padding='SAME',
				bias_attr=use_bias,
			)

def transposed_convolution_3D(in_channels, filters, kernel_size, strides, use_bias):
	"""Performs 3D transposed convolution without activation function.
	If followed by batch normalization, set use_bias=False.
	"""
	return nn.Conv3DTranspose(
                in_channels=in_channels,
				out_channels=filters,
				kernel_size=kernel_size,
				stride=strides,
				padding='SAME',
				bias_attr=use_bias,
				data_format="NDHWC"
			)

def batch_norm(inputs, training, name=None):
	if conf_basic_ops['use_batch_norm']:
		return paddle.fluid.layers.batch_norm(
					input=inputs,
					momentum=conf_basic_ops['momentum'],
					epsilon=conf_basic_ops['epsilon'],
                    is_test=bool(1-training),
					name=name
				)
	else:
		return inputs

def relu(inputs,name=None):
	return nn.ReLU(name=name)(inputs) if conf_basic_ops['relu_type'] == 'relu' else nn.ReLU6(name=name)(inputs)