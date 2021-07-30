import paddle
import os
import tifffile
import sys
import numpy as np

from unet.network import UNet, ProjectionNet, get_loss
from unet.data import train_input_function, pred_input_function, load_testing_tiff
from unet.data import PercentileNormalizer, PadAndCropResizer, PatchPredictor
from paddle.static import InputSpec
"""This script trains or evaluates the model.
"""

class Model(object):

	def __init__(self, opts, conf_unet):
		self.opts = opts
		self.conf_unet = conf_unet
		
	def train(self):
		print('start training ... ')
		model = UNet(self.conf_unet)
		model.train()
		optimizer = paddle.optimizer.Adam(
					learning_rate=self.opts.learning_rate,
					beta1=0.5,
					beta2=0.999,
					parameters=model.parameters())
		epoch_num = self.opts.num_iters * self.opts.batch_size // self.opts.num_train_pairs

		train_loader = train_input_function(self.opts)
		for epoch in range(epoch_num):
			for batch_id, data in enumerate(train_loader):
				x_data, y_data = data
				img = paddle.to_tensor(x_data)
				label = paddle.to_tensor(y_data)
				outputs, penult = model(img, training=True)
				loss = get_loss(label, outputs, penult, self.opts.loss_type,
					self.opts.probalistic, self.opts.offset, self.conf_unet['dimension'])
				avg_loss = paddle.mean(loss)
				print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, batch_id, avg_loss.numpy()))
				if batch_id % 10 == 0:
					paddle.save(model.state_dict(), self.opts.model_dir + 'checkpoint_%s' %batch_id + 'palm.pdparams')
					paddle.save(optimizer.state_dict(), self.opts.model_dir + 'checkpoint_%s' %batch_id + 'palm.pdopt')

				avg_loss.backward()
				optimizer.step()
				optimizer.clear_grad()

	def predict(self):
		print('start predicting ... ')
		sources, fnames = load_testing_tiff(self.opts.tiff_dataset_dir, self.opts.num_test_pairs)
		pred_result_dir = os.path.join(self.opts.result_dir, 'checkpoint_%s' % self.opts.checkpoint_num)
		if not os.path.exists(pred_result_dir):
			os.makedirs(pred_result_dir)
		else:
			print('The result dir for checkpoint_num %s already exist.' % self.opts.checkpoint_num)
			# return 0

		# Using the Winograd non-fused algorithms provides a small performance boost.
		
		model = UNet(self.conf_unet)

		layer_state_dict = paddle.load(self.opts.model_dir +'checkpoint_%s' %self.opts.checkpoint_num + "palm.pdparams")
		# opt_state_dict = paddle.load(self.opts.model_dir + 'checkpoint_%s' %self.opts.checkpoint_num +"adam.pdopt")
		model.set_state_dict(layer_state_dict)
		resizer = PadAndCropResizer()
		cropper = PatchPredictor(self.opts.predict_patch_size, self.opts.overlap, self.opts.proj_model) if \
					self.opts.cropped_prediction else None
		normalizer = PercentileNormalizer() if self.opts.CARE_normalize else None
		div_n = (4 if self.opts.proj_model else 2)**(self.conf_unet['depth']-1)
		excludes = ([3,0], 2) if self.opts.proj_model else (3,3)

		for idx, source in enumerate(sources):
			print('Predicting testing sample %d, shape %s ...' % (idx, str(source.shape)))
			source = normalizer.before(source, 'ZYXC') if self.opts.CARE_normalize else source
			source = resizer.before(source, div_n=div_n, exclude=excludes[0])
			prediction = []
			if self.opts.cropped_prediction:
				patches = cropper.before(source, div_n)
				input_fn=pred_input_function(self.opts, patches)
				for data in input_fn:
					img = paddle.to_tensor(data)
					prediction.append(model(img, training=False)[0])
				prediction = cropper.after(np.array(prediction))
			else:
				# Take the entire image as the input and make predictions.
				# If the image is very large, set --gpu_id to -1 to use cpu mode.
				input_fn=pred_input_function(self.opts, source[None])
				for data in input_fn:
					print(data.shape)
					img = paddle.to_tensor(data)[:,:32,:64,:64,:]
					pred = model(img, training=True)[0]
					prediction.append(pred)

			prediction = np.array(prediction)
			# prediction = resizer.after(prediction, exclude=excludes[1])
			# prediction = normalizer.after(prediction) if \
			# 				self.opts.CARE_normalize and normalizer.do_after() else prediction
			# prediction = prediction[0] if self.opts.proj_model else prediction
			path_tiff = os.path.join(pred_result_dir, 'pred_'+fnames[idx])
			tifffile.imsave(path_tiff, prediction[..., 0])
			print('saved:', path_tiff)

		print('Done.')
		sys.exit(0)
