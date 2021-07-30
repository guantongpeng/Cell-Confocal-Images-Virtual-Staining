import os
import numpy as np
import fnmatch
from paddle.fluid.dygraph.nn import Pool2D
from tifffile import imread
import paddle
import random


def load_training_npz(npz_dataset_dir, num_train_pairs):
	""" Loading data from npz files.
	DTYPE: np.float32
	FORMAT options:
		A single npz file containing all training data:
			{'X': (n_sample, n_channel, (depth,) height, width),
			 'Y': (n_sample, n_channel, (depth,) height, width)}
		Multiple npz files where each one contains one training sample:
			NOTE: (depth,) height, width) can vary for different samples.
			{'X': (n_channel, (depth,) height, width),
			 'Y': (n_channel, (depth,) height, width)}
	Return:
		A single npz file containing all training data:
			sources: An numpy array of shape [n_sample, (depth,) height, width, n_channel]
			targets: An numpy array of shape [n_sample, (depth,) height, width, n_channel]
		Multiple npz files where each one contains one training sample:
			NOTE: (depth,) height, width) can vary for different samples.
			sources: A list of numpy arrays of shape [(depth,) height, width, n_channel]
			targets: A list of numpy arrays of shape [(depth,) height, width, n_channel]
	"""
	print("Loading npz file(s)...")
	fnames = [fname for fname in os.listdir(npz_dataset_dir) if 
				fnmatch.fnmatch(fname, '*.npz')]
	fnames.sort()
	fnames = [os.path.join(npz_dataset_dir, fname) for fname in fnames]

	if len(fnames)==1:
		data = np.load(fnames[0])
		# moving channel dimension to the last
		sources = np.moveaxis(data['X'], 1, -1)
		targets = np.moveaxis(data['Y'], 1, -1)

	else:
		sources = []
		targets = []

		for fname in fnames:
			data = np.load(fname)
			sources.append(np.moveaxis(data['X'], 0, -1))
			targets.append(np.moveaxis(data['Y'], 0, -1))
		
		assert len(sources) == num_train_pairs, "len(sources) is %d" % len(sources)

	print("Data loaded.")
	return sources, targets


def load_testing_tiff(tiff_dataset_dir, num_test_pairs):
	""" Loading data from tiff files.
	DTYPE: np.float32
	Return:
		A list of numpy arrays of shape [(depth,) height, width, n_channel].
		Each entry in the list corresponds to one data sample.
	"""
	print("Loading tiff file(s)...")
	fnames = [fname for fname in os.listdir(tiff_dataset_dir) if 
				fnmatch.fnmatch(fname, '*.tif*')]
	fnames.sort()
	fpaths = [os.path.join(tiff_dataset_dir, fname) for fname in fnames]

	sources = []

	for fpath in fpaths:
		data = imread(fpath)
		data = np.expand_dims(data, axis=-1)
		sources.append(data)
	
	assert len(sources) == num_test_pairs, "len(sources) is %d" % len(sources)

	print("Data loaded.")
	return sources, fnames

def data_generator(x, y, batch_size, shuffle):
    imgs_length = len(x)
    # 定义数据集每个数据的序号，根据序号读取数据
    index_list = list(range(imgs_length))
    if shuffle:
            # 训练模式下打乱数据
            random.shuffle(index_list)
    imgs_list = []
    labels_list = []
    for i in index_list:
        # 将数据处理成希望的类型
        img = np.array(x[i]).astype('float32')
        label = np.array(y[i]).astype('float32')
        imgs_list.append(img) 
        labels_list.append(label)
        if len(imgs_list) == batch_size:
            # 获得一个batchsize的数据，并返回
            yield np.array(imgs_list), np.array(labels_list)
            # 清空数据读取列表
            imgs_list = []
            labels_list = []
    # 如果剩余数据的数目小于BATCHSIZE，
    # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
    if len(imgs_list) > 0:
        yield np.array(imgs_list), np.array(labels_list)


def data_crop(opts, sources, targets, shuffle=True):
	source_patch, target_patch = [], []
	for i in range(opts.num_iters * opts.batch_size):
		if shuffle:
			idx = np.random.randint(len(sources))
		else:
			idx = i // len(sources)
		source, target = sources[idx], targets[idx]

		# random crop
		# TODO: compatability to 2D inputs
		valid_shape = source.shape[:-1] - np.array(opts.train_patch_size)
		z = np.random.randint(0, valid_shape[0])
		x = np.random.randint(0, valid_shape[1])
		y = np.random.randint(0, valid_shape[2])
		s = (slice(z, z+opts.train_patch_size[0]), 
				slice(x, x+opts.train_patch_size[1]), 
				slice(y, y+opts.train_patch_size[2]))
		source_patch.append(source[s])
		target_patch.append(target[s])
	return source_patch, target_patch

def train_input_function(opts):
	sources, targets = load_training_npz(opts.npz_dataset_dir, opts.num_train_pairs)

	# if opts.already_cropped:
	# 	# The training data have been cropped.
	# 	# The training data are stored in a single npz file.
	# 	# repeats = opts.num_iters * opts.batch_size // opts.num_train_pairs
	# 	input_fn = data_generator(
	# 		x=sources, y=targets,
	# 		batch_size=opts.batch_size,
	# 		shuffle=True)

	# elif not opts.save_tfrecords:
		# The training data have NOT been cropped.
		# The training data are stored in multiple npz files,
		# where each one contains one training sample.
	source_patch, target_patch = data_crop(opts, sources, targets, shuffle=True)
	input_fn = data_generator(x=source_patch, y=target_patch,
			batch_size=opts.batch_size,
			shuffle=True)

	# else:
	# 	# Using tfrecords can handle both cases.
	# 	save_tfrecord(opts, sources, targets)
	# 	input_fn = lambda: input_fn_tfrecord(opts)
	return input_fn


def pred_input_function(opts, sources):

	return pred_data_generator(
		x=sources,
		batch_size=opts.test_batch_size if opts.cropped_prediction else 1,
		shuffle=False)
		
def pred_data_generator(x, batch_size, shuffle):
    imgs_length = len(x)
    # 定义数据集每个数据的序号，根据序号读取数据
    index_list = list(range(imgs_length))
    if shuffle:
            # 训练模式下打乱数据
            random.shuffle(index_list)
    imgs_list = []
    labels_list = []
    for i in index_list:
        # 将数据处理成希望的类型
        img = np.array(x[i]).astype('float32')
        imgs_list.append(img) 
        if len(imgs_list) == batch_size:
            # 获得一个batchsize的数据，并返回
            yield np.array(imgs_list)
            # 清空数据读取列表
            imgs_list = []
    # 如果剩余数据的数目小于BATCHSIZE，
    # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
    if len(imgs_list) > 0:
        yield np.array(imgs_list)