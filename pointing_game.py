import pickle
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset.dataset import load_data
from dataset.utils import *
import matplotlib.pyplot as plt
from models.vgg import *
import lrp
from models.converter import convert
import argparse
from captum.attr import IntegratedGradients
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import math
from tqdm import tqdm

torch.set_printoptions(sci_mode=False)


# Joint Model
def ModelXtoCtoY(n_class_attr, num_classes, n_attributes, expand_dim, use_relu, use_sigmoid):
	vgg_model = vgg16_bn(pretrained=False, num_classes=num_classes, n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim, train=True)
	model1 = x_to_c_model(freeze=False, model=vgg_model)
	if n_class_attr == 3:
		model2 = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim, train=True)
	else:
		model2 = MLP(input_dim=n_attributes, num_classes=num_classes, expand_dim=expand_dim, train=train)
	return End2EndModel(model1, model2, use_relu, use_sigmoid, n_class_attr)


# https://github.com/fhvilshoj/TorchLRP/blob/74253a1be05f0be0b7c535736023408670443b6e/examples/visualization.py#L60
def heatmap(X, cmap_name="seismic"):
	cmap = plt.cm.get_cmap(cmap_name)

	if X.shape[1] in [1, 3]:  # move channel index to end + convert to np array
		X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
	if isinstance(X, torch.Tensor):  # convert tensor to np array
		X = X.detach().cpu().numpy()

	shape = X.shape
	tmp = X.sum(axis=-1) # Reduce channel axis

	tmp = project(tmp, output_range=(0, 255)).astype(int)
	tmp = cmap(tmp.flatten())[:, :3].T
	tmp = tmp.T

	shape = list(shape)
	shape[-1] = 3
	return tmp.reshape(shape).astype(np.float32)


def reduce_heatmap(X, cmap_name="seismic"):
	"""
	reduce dimensions to 1 x W x H
	"""
	cmap = plt.cm.get_cmap(cmap_name)

	if X.shape[1] in [1, 3]:  # move channel index to end + convert to np array
		X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
	if isinstance(X, torch.Tensor):  # convert tensor to np array
		X = X.detach().cpu().numpy()

	shape = X.shape
	tmp = X.sum(axis=-1) # Reduce channel axis
	return tmp.astype(np.float32)


def project(X, output_range=(0, 1)):
	absmax = np.abs(X).max(axis=tuple(range(1, len(X.shape))), keepdims=True)
	X /= absmax + (absmax == 0).astype(float)
	X = (X+1) / 2. # range [0, 1]
	X = output_range[0] + X * (output_range[1] - output_range[0]) # range [x, y]
	return X


def enumerate_data(models, data_loader, output_path, class_index_to_string, concept_index_to_string, n_concepts=112, full_model=None, device="cpu", rules=None, mode="lrp", sample_counter=[]):
	"""
	enumerate entire dataset

	args:
		models (list): List of pytorch models
		data_loader: Pytorch data loader
		output_path (str): Path to save results
		class_index_to_string (dict): dictionary to convert ints to strings (classes to class names)
		concept_index_to_string (dict): dictionary to convert ints to strings (concepts to concept names)
		n_concepts (int): Number of concepts in dataset
		full_model: pytorch model. Used to get class prediction before results are generated
		device (str): Device to generate results on (e.g. cpu or cuda:0)
		rules (list): list of LRP rules (from TorchLRP). Number of rules must match number of models in list
		mode (str): Type of results to generate
		sample_counter (list of ints): list of ints. Lengh should match number of classes in dataset
	"""

	distances = {}

	for images, labels, concepts, locs in tqdm(data_loader, unit="batches"):

		predictions = full_model(images.to(device))

		# itterate over predicted concepts
		for idx, concept_prediction in enumerate(predictions[1]):

			# only generate explanation if number of samples for class is less than max to generate or -1 (all samples)
			if sample_counter[labels[idx].item()] < args.samples_per_class or args.samples_per_class == -1:
				sample_counter[labels[idx].item()] += 1

				# get concept predictions after sig
				pred_concepts = torch.nn.Sigmoid()(concept_prediction)

				# Only continue for concept predicted as precent
				for idy, i in enumerate(pred_concepts):
					if i.item() >= 0.5:  # only continue for concepts predicted as present

						target_locs = conceptToBirstPart(idy)

						# if visible parts is over 1 then skip as we cannot know if all points are relevant
						# with current version of pointing game and thus this may affect the average di
						if len(target_locs) > 1:
							num_visible = 0
							for j in target_locs:
								if locs[0][j][0].item() != 0.0 and locs[0][j][1].item() != 0.0:
									num_visible += 1
							if num_visible > 1:
								continue

						# only continue if bird part related to concept is visible (x != 0 and Y != 0)
						for j in target_locs:
							if locs[0][j][0].item() != 0.0 and locs[0][j][1].item() != 0.0:

								if mode == "LRP":
									distance = get_LRP_distance(models, rules, images[idx].to(device), idy, locs[0][j])
								elif mode == "IG":
									distance = get_IG_distance(models[0], images[idx].to(device), idy, locs[0][j])
								else:
									print("No mode selected")



								if j in distances:
									distances_new = distances[j]
									distances_new.append(distance)
								else:
									distances_new = [distance]
								distances[j] = distances_new

	if not os.path.exists(output_path):
		try:
			os.makedirs(output_path)
		except OSError:
			print(f"Creation of the directory {output_path} failed")
		else:
			pass

	all_distances = []
	with open(f'{output_path}/pointing_game.txt', 'w') as f:
		for key in distances:
			all_distances = all_distances + distances[key]
			f.write(f'Part {key + 1} average distance: {sum(distances[key]) / len(distances[key])}\n')
		f.write(f'All parts average distance: {sum(all_distances) / len(all_distances)}\n')


def get_LRP_distance(models, rules, image, concept_no, true_loc):
	"""
	given a data sample, concept ID and location X and Y generate a saliecy map and return the distance between the most salient point and the given location

	args:
		models (list): List of pytorch models
		rules (list): list of LRP rules (from TorchLRP). Number of rules must match number of models in list
		image (tensor): single input image
		concept_no (int): index of concept to generate saliency map for
		true_loc (tuple): tuple of ints in the format (x, y)
	"""
	image = image.unsqueeze(0)
	image.requires_grad_(True)
	image.grad = None  # Reset gradient

	# Loop over models in list (used for composite rule)
	# if len(models) == 1 then a single rule will be used, else composite rules
	input = None
	for idx, model in enumerate(models):
		# If image has not been passed through a model yet
		if input is None:
			input = image
		pred_concepts = model.forward(input, explain=True, rule=rules[idx])
		input = pred_concepts

	# get saliecy maps for selected concept
	filter_out = torch.zeros_like(pred_concepts)
	filter_out[:, concept_no] += 1

	# Get the gradient of each input
	image_gradient = torch.autograd.grad(
		pred_concepts,
		image,
		grad_outputs=filter_out,
		retain_graph=True)[0]

	attr = reduce_heatmap(image_gradient, cmap_name='seismic')
	top_point = np.unravel_index(attr.argmax(), attr.shape)

	#attr_heat = heatmap(image_gradient, cmap_name='seismic')

	a = true_loc[0].item() - top_point[2]
	b = true_loc[1].item() - top_point[1]

	#plt.imshow(attr_heat.squeeze(), cmap='seismic')
	#plt.show()
	#plt.imshow(attr_heat.squeeze(), cmap='seismic')
	#plt.plot([true_loc[0].item()], [true_loc[1].item()], marker='o', markersize=3, color="green")
	#plt.plot([top_point[2]], [top_point[1]], marker='o', markersize=3, color="purple")
	#plt.show()

	return math.sqrt((a * a) + (b * b))  # return distance


def get_IG_distance(nt, image, concept_no, true_loc):
	"""
	given a data sample, generate saliency maps with IG
	"""

	image = image.unsqueeze(0)
	image.requires_grad_(True)
	image.grad = None  # Reset gradient
	baselines = image * 0

	attributions_ig = nt.attribute(
		image,
		target=concept_no,
		baselines=baselines,
		nt_type='smoothgrad_sq',
		nt_samples=25,
		internal_batch_size=25,
		stdevs=0.2
	)

	attr = heatmap(attributions_ig, cmap_name='seismic')

	top_point = np.unravel_index(attr.argmax(), attr.shape)

	#attr_heat = heatmap(image_gradient, cmap_name='seismic')

	a = true_loc[0].item() - top_point[2]
	b = true_loc[1].item() - top_point[1]

	#plt.imshow(attr_heat.squeeze(), cmap='seismic')
	#plt.show()
	#plt.imshow(attr_heat.squeeze(), cmap='seismic')
	#plt.plot([true_loc[0].item()], [true_loc[1].item()], marker='o', markersize=3, color="green")
	#plt.plot([top_point[2]], [top_point[1]], marker='o', markersize=3, color="purple")
	#plt.show()

	return math.sqrt((a * a) + (b * b))  # return distance


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Train and test a Concept Bottleneck Model')
	parser.add_argument(
		'--model',
		type=str,
		help='XtoC or joint model path'
	)
	parser.add_argument(
		'--rules',
		nargs='+',
		help='LRP rules defined in TorchLRP'
	)
	parser.add_argument(
		'--use_sigmoid',
		action='store_true',
		help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model'
	)
	parser.add_argument(
		'--use_relu',
		action='store_true',
		help='Whether to include relu activation before using attributes to predict Y. For end2end & bottleneck model'
	)
	parser.add_argument(
		'--expand_dim',
		type=int,
		help='Size of middle layer in CtoY model',
		default=0
	)
	parser.add_argument(
		'--n_class_attr',
		type=int,
		help='whether attr prediction is a binary or triary classification',
		default=2
	)
	parser.add_argument(
		'--n_concepts',
		type=int,
		help='number of concepts',
		default=112
	)
	parser.add_argument(
		'--n_classes',
		type=int,
		help='number of concepts',
		default=200
	)
	parser.add_argument(
		'--output',
		type=str,
		help='output directory path'
	)
	parser.add_argument(
		'--mode',
		type=str,
		default="LRP",
		choices=["IG", "LRP"],
		help='mode to run script'
	)
	parser.add_argument(
		'--samples_per_class',
		type=int,
		help='Max number of samples to generate results for per classification. -1 to run all',
		default=10
	)
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		print("Device:", device, torch.cuda.get_device_name(0))
	else:
		print("Device:", device)

	# load model and dataset
	test_loader = load_data("./dataset/CUB/dataset_splits/CBM_dataset_split/val.pkl", 16, image_dir='dataset/CUB/data/images', return_locs=True)

	class_index_to_string = IndexToString("./dataset/CUB/data/classes.txt", classes=True)
	concept_index_to_string = IndexToString("./dataset/CUB/dataset_splits/CBM_dataset_split/attributes.txt")

	XtoCtoY_model = ModelXtoCtoY(n_class_attr=args.n_class_attr, num_classes=args.n_classes, n_attributes=args.n_concepts, expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid)
	XtoCtoY_model.load_state_dict(torch.load(args.model))
	XtoCtoY_model.to(device)
	XtoCtoY_model.eval()

	output_path = f"{args.output}/{args.mode}"

	# create list to count number of samples have results generated for per class
	sample_counter = []
	for i in range(args.n_classes):
		sample_counter.append(0)

	if args.mode == "LRP":
		XtoC_model = convert(XtoCtoY_model.first_model)

		XtoC_model_1 = []
		XtoC_model_2 = []
		XtoC_model_3 = []

		# define LRP rules for each layer of model
		for idx, m in enumerate(XtoC_model.children()):
			if idx < 18:
				XtoC_model_1.append(m)
			elif idx < 29:
				XtoC_model_2.append(m)
			else:
				XtoC_model_3.append(m)

		XtoC_model_1 = lrp.Sequential(*XtoC_model_1)
		XtoC_model_2 = lrp.Sequential(*XtoC_model_2)
		XtoC_model_3 = lrp.Sequential(*XtoC_model_3)

		XtoC_model_1.to(device)
		XtoC_model_2.to(device)
		XtoC_model_3.to(device)
		XtoC_model_1.eval()
		XtoC_model_2.eval()
		XtoC_model_3.eval()

		model_parts = [XtoC_model_1, XtoC_model_2, XtoC_model_3]
		print("model loaded")

		enumerate_data(model_parts, test_loader, output_path, class_index_to_string, concept_index_to_string, args.n_concepts, XtoCtoY_model, device, rules=args.rules, mode=args.mode, sample_counter=sample_counter)
	elif args.mode == "IG":
		XtoC_model = XtoCtoY_model.first_model
		XtoC_model.to(device)
		XtoC_model.eval()
		ig = IntegratedGradients(XtoC_model)
		nt = NoiseTunnel(ig)
		print("model loaded")

		enumerate_data([nt], test_loader, output_path, class_index_to_string, concept_index_to_string, args.n_concepts, XtoCtoY_model, device, mode=args.mode, sample_counter=sample_counter)
	else:
		print("mode not recognised")
