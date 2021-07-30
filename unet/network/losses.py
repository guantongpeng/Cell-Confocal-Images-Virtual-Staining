import paddle
from paddle import nn
import paddle.nn.functional as F
from .basic_ops import *

def get_loss(labels, preds, penult, loss_type, probalistic, offset, dimension):
	"""
	Args:
		labels: The ground truth output tensor, same dimensions as 'inputs' and 'outputs'.
		preds: Output tensor of UNet.
		penult: Penultimate layer.
		loss_type: String. Type of loss function. Can be either 'MSE' (Mean Squared Error) or 'MAE' (Mean Absolute Error).
		probalistic: Boolean. Whether to use probalistic loss.
		offset: Boolean. Whether to add UNet inputs to outputs.
		dimension: String. Dimension of image, '3D' or '2D'.
	"""

	if dimension == '2D':
		convolution = convolution_2D
	elif dimension == '3D':
		convolution = convolution_3D

	if loss_type == 'MSE':
		if probalistic:
			sigma = convolution(penult, 1, 1, 1, False)
			sigma = paddle.fluid.layers.softplus(sigma) + 1e-3
			loss = paddle.fluid.layers.reduce_mean(
                paddle.fluid.layers.elementwise_div(paddle.square(preds-labels), 2*sigma**2) 
                + paddle.log(sigma))
		else:
			loss = F.mse_loss(labels, preds)

	elif loss_type == 'MAE':
		if probalistic:
			sigma = convolution(penult, 1, 1, 1, False)
			sigma = paddle.fluid.layers.softplus(sigma) + 1e-3
			loss = paddle.fluid.layers.reduce_mean(
                paddle.fluid.layers.elementwise_div(paddle.abs(preds-labels), sigma) 
                + paddle.log(sigma))
		else:
			loss = F.l1_loss(labels, preds)

	else:
		raise ValueError("The loss_type (%s) must be MSE or MAE." % (loss_type))

	return loss