from .basic_ops import *
import paddle
import numpy as np

"""This script defines 2D and 3D multihead self-attention layers.
"""

def self_attention(inputs, total_key_filters, total_value_filters, output_filters,
		num_heads, training, dimension, layer_type, name, dropout_rate=0.0, use_softmax=True,
		use_bias=None):
	
	if total_key_filters % num_heads != 0:
		raise ValueError("Key depth (%d) must be divisible by the number of "
						"attention heads (%d)." % (total_key_filters, num_heads))
	if total_value_filters % num_heads != 0:
		raise ValueError("Value depth (%d) must be divisible by the number of "
						"attention heads (%d)." % (total_value_filters, num_heads))
	if layer_type not in ['SAME', 'DOWN', 'UP', 'UP4']:
		raise ValueError("Layer type (%s) must be one of SAME, DOWN, UP, UP4." % (layer_type))

	if dimension == '2D':
		compute_qkv = compute_qkv_2d
		split_heads = split_heads_2d
		unfold = unfold_2d
		combine_heads = combine_heads_2d
		output_transfrom = convolution_2D
	elif dimension == '3D':
		compute_qkv = compute_qkv_3d
		split_heads = split_heads_3d
		unfold = unfold_3d
		combine_heads = combine_heads_3d
		output_transfrom = convolution_3D
	else:
		raise ValueError("Dimension (%s) must be 2D or 3D." % (dimension))

	with paddle.static.name_scope(name):
		# produce q, k, v
		q, k, v = compute_qkv(inputs, total_key_filters, total_value_filters, use_bias,
					layer_type)

		# after splitting, shape is [batch, heads, d, h, w, channels / heads]
		q_split = split_heads(q, num_heads)
		k_split = split_heads(k, num_heads)
		v_split = split_heads(v, num_heads)
		
		# normalization recommended by "Attention is All You Need"
		key_filters_per_head = total_key_filters // num_heads
		q_split *= key_filters_per_head**-0.5

		output_shape = list(np.array(q_split.shape[0:-1] + [v_split.shape[-1]]))

		# flatten q,k,v
		q_new = unfold(q_split)
		k_new = unfold(k_split)
		v_new = unfold(v_split)

		# attention
		o = dot_product_attention(q_new, k_new, v_new, training, dropout_rate, use_softmax)

		# putting the representations back in the right place
		o = paddle.reshape(o, output_shape)

		# combine heads and perform output transformation
		o = combine_heads(o)

		o = output_transfrom(o.shape[-1], output_filters, 1, 1, use_bias)(o)

		return o, q


def compute_qkv_2d(inputs, total_key_filters, total_value_filters, use_bias, layer_type):
	"""Perform qkv transformations and compute query, key and value.
	Args:
		inputs: a Tensor with shape [batch, h, w, channels]
		total_key_filters: an integer
		total_value_filters: an integer
		use_bias: a boolean deciding whether to use the bias term in qkv transformations
		layer_type: a string, type of this layer -- SAME, DOWN, UP
	
	Returns:
		q: a Tensor with shape [batch, _h, _w, total_key_filters]
		k: a Tensor with shape [batch, h, w, total_key_filters]
		v: a Tensor with shape [batch, h, w, total_value_filters]
	"""
	# transformation for q
	if layer_type == 'SAME':
		q = convolution_2D(inputs, total_key_filters, 1, 1, use_bias)
	elif layer_type == 'DOWN':
		q = convolution_2D(inputs, total_key_filters, 3, 2, use_bias)
	elif layer_type == 'UP':
		q = transposed_convolution_2D(inputs, total_key_filters, 3, 2, use_bias)

	# linear transformation for k
	k = convolution_2D(inputs, total_key_filters, 1, 1, use_bias)

	# linear transformation for v
	v = convolution_2D(inputs, total_value_filters, 1, 1, use_bias)

	return q, k, v


def compute_qkv_3d(inputs, total_key_filters, total_value_filters, use_bias, layer_type):
	"""Perform qkv transformations and compute query, key and value.
	Args:
		inputs: a Tensor with shape [batch, d, h, w, channels]
		total_key_filters: an integer
		total_value_filters: an integer
		use_bias: a boolean deciding whether to use the bias term in qkv transformations
		layer_type: a string, type of this layer -- SAME, DOWN, UP
	
	Returns:
		q: a Tensor with shape [batch, _d, _h, _w, total_key_filters]
		k: a Tensor with shape [batch, d, h, w, total_key_filters]
		v: a Tensor with shape [batch, d, h, w, total_value_filters]
	"""
	# transformation for q
	if layer_type == 'SAME':
		q = convolution_3D(inputs.shape[-1], total_key_filters, 1, 1, use_bias)(inputs)
	elif layer_type == 'DOWN':
		q = convolution_3D(inputs.shape[-1], total_key_filters, 3, 2, use_bias)(inputs)
	elif layer_type == 'UP':
		q = transposed_convolution_3D(inputs.shape[-1], total_key_filters, 3, 2, use_bias)(inputs)
	# ProjectionNet uses 4 times up-sampling and down-sampling. For projection models only, e.g. Flywing Projections.
	elif layer_type == 'UP4':
		q = paddle.reshape(inputs, paddle.concat([paddle.shape(inputs)[0:1]*paddle.shape(inputs)[1:2], paddle.shape(inputs)[2:]],0))
		q = paddle.fluid.layers.resize_nearest(q, paddle.concat([paddle.shape(inputs)[2:3]*4, paddle.shape(inputs)[3:4]*4],0))
		q = paddle.reshape(q, paddle.concat([paddle.shape(inputs)[:2], paddle.shape(q)[1:]], 0))
		
	# linear transformation for k
	k = convolution_3D(inputs.shape[-1], total_key_filters, 1, 1, use_bias)(inputs)

	# linear transformation for v
	v = convolution_3D(inputs.shape[-1], total_value_filters, 1, 1, use_bias)(inputs)

	return q, k, v


def reshape_range(tensor, i, j, shape):
	"""Reshapes a tensor between dimensions [i,j)."""
	target_shape = [paddle.shape(tensor)[:i]] + shape + [paddle.shape(tensor)[j:]]
			
	return paddle.reshape(tensor, target_shape)


def unfold_2d(x):
	x_shape = paddle.shape(x)
	# [batch, heads, length, channels], length = h*w
	x = reshape_range(x, 2, 4, [paddle.prod(x_shape[2:4])])
	return x


def unfold_3d(x):
	x_shape = paddle.shape(x)
	# [batch, heads, length, channels], length = d*h*w
	x = reshape_range(x, 2, 5, list(np.array(paddle.prod(x_shape[2:5]))))
	return x


def dot_product_attention(q, k, v, training, dropout_rate, use_softmax):
	"""Dot-product attention.
	Args:
		q: a Tensor with shape [batch, heads, length_q, channels_k]
		k: a Tensor with shape [batch, heads, length_kv, channels_k]
		v: a Tensor with shape [batch, heads, length_kv, channels_v]
		training: a boolean for dropout
		dropout_rate: a float between 0.0 and 1.0. No dropout if dropout_rate = 0.0
		use_softmax: a boolean deciding whether to use softmax. Note that
			use_softmax = False will automatically set dropout_rate = 0.0
	Returns:
		A Tensor with shape [batch, heads, length_q, channels_v]
	"""
	if use_softmax:
		# [batch, num_heads, length_q, length_kv]
		attention_weights = paddle.matmul(q, k, transpose_y=True)

		# normalize attention
		attention_weights = paddle.nn.Softmax(attention_weights)

		# dropping out the attention links for each of the heads
		if dropout_rate != 0.0:
			attention_weights = paddle.nn.functional.dropout(attention_weights, dropout_rate, training)

		return paddle.matmul(attention_weights, v)
	else:
		# To save computation, compute the multiplication between K^T and V first.
		kv = paddle.matmul(k, v, transpose_x=True)

		# normalize
		kv = kv/paddle.cast(paddle.shape(q)[2], paddle.float32)

		return paddle.matmul(q, kv)


def split_heads_2d(x, num_heads):
	"""Split channels (last dimension) into multiple heads (becomes dimension 1).
	
	Args:
		x: a Tensor with shape [batch, h, w, channels]
		num_heads: an integer
	
	Returns:
		a Tensor with shape [batch, num_heads, h, w, channels / num_heads]
	"""
	return paddle.transpose(split_last_dimension(x, num_heads), [0, 3, 1, 2, 4])


def split_heads_3d(x, num_heads):
	"""Split channels (last dimension) into multiple heads (becomes dimension 1).
	
	Args:
		x: a Tensor with shape [batch, d, h, w, channels]
		num_heads: an integer
	
	Returns:
		a Tensor with shape [batch, num_heads, d, h, w, channels / num_heads]
	"""
	return paddle.transpose(split_last_dimension(x, num_heads), [0, 4, 1, 2, 3, 5])


def split_last_dimension(x, n):
	"""Reshape x so that the last dimension becomes two dimensions.
	The first of these two dimensions is n.
	Args:
		x: a Tensor with shape [..., m]
		n: an integer.
	Returns:
		a Tensor with shape [..., n, m/n]
	"""
	old_shape = x.shape
	last = old_shape[-1]
	new_shape = old_shape[:-1] + [n] + [last // n if last else None]
	ret = paddle.reshape(x, old_shape[:-1] + [n, -1])
	ret.reshape(new_shape)
	return ret


def combine_heads_2d(x):
	"""Inverse of split_heads_2d.
	Args:
		x: a Tensor with shape [batch, num_heads, h, w, channels / num_heads]
	Returns:
		a Tensor with shape [batch, h, w, channels]
	"""
	return combine_last_two_dimensions(paddle.transpose(x, [0, 2, 3, 1, 4]))


def combine_heads_3d(x):
	"""Inverse of split_heads_3d.
	Args:
		x: a Tensor with shape [batch, num_heads, d, h, w, channels / num_heads]
	Returns:
		a Tensor with shape [batch, d, h, w, channels]
	"""
	return combine_last_two_dimensions(paddle.transpose(x, [0, 2, 3, 4, 1, 5]))


def combine_last_two_dimensions(x):
	"""Reshape x so that the last two dimension become one.
	Args:
		x: a Tensor with shape [..., a, b]
	Returns:
		a Tensor with shape [..., a*b]
	"""
	old_shape = x.shape
	a, b = old_shape[-2:]
	new_shape = old_shape[:-2] + [a * b if a and b else None]
	ret = paddle.reshape(x, x.shape[:-2] + [-1])
	ret.reshape(new_shape)
	return ret
