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

torch.set_printoptions(sci_mode=False)

# A single figure is defined as putting this in a loop quickly uses up all system memory.
# This figure is reused for all plots
#FIG = plt.figure()
#AX = FIG.add_subplot(111)
FIG, AX = plt.subplots()


# Joint Model
def ModelXtoCtoY(n_class_attr, num_classes, n_attributes, expand_dim, use_relu, use_sigmoid, train=True, **kwargs):
	vgg_model = vgg16_bn(pretrained=False, num_classes=num_classes, n_attributes=n_attributes, bottleneck=True, expand_dim=expand_dim, train=train, **kwargs)
	model1 = x_to_c_model(freeze=False, model=vgg_model)
	if n_class_attr == 3:
		model2 = MLP(input_dim=n_attributes * n_class_attr, num_classes=num_classes, expand_dim=expand_dim, train=train)
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


def project(X, output_range=(0, 1)):
	absmax = np.abs(X).max(axis=tuple(range(1, len(X.shape))), keepdims=True)
	X /= absmax + (absmax == 0).astype(float)
	X = (X+1) / 2. # range [0, 1]
	X = output_range[0] + X * (output_range[1] - output_range[0]) # range [x, y]
	return X


def model_class_pred(model, image, device):
	image = image.unsqueeze(0)
	image.requires_grad_(True)
	image.grad = None

	# get model prediction and LRP map for classification
	prediction = model(image)
	return (prediction[0].max(1, keepdim=True)[1]).item()


def enumerate_data(models, data_loader, output_path, class_index_to_string, concept_index_to_string, n_concepts=112, device="cpu", full_model=None, rules=None, mode="singlar", sample_counter=[]):
	"""
	enumerate entire dataset

	args:
		models (list): List of pytorch models
		data_loader: Pytorch data loader
		output_path (str): Path to save results
		class_index_to_string (dict): dictionary to convert ints to strings (classes to class names)
		concept_index_to_string (dict): dictionary to convert ints to strings (concepts to concept names)
		n_concepts (int): Number of concepts in dataset
		device (str): Device to generate results on (e.g. cpu or cuda:0)
		full_model: pytorch model. Used to get class prediction before results are generated
		rules (list): list of LRP rules (from TorchLRP). Number of rules must match number of models in list
		mode (str): Type of results to generate
		sample_counter (list of ints): list of ints. Lengh should match number of classes in dataset
	"""
	sample_count = 0  # keep track of samples. Used to ensure we do not save two samples with the same name
	for images, labels, concepts, orig_image in data_loader:
		for idx, image in enumerate(images):

			# only generate explanation if number of samples for class is less than max to generate
			if sample_counter[labels[idx].item()] < args.samples_per_class:
				sample_counter[labels[idx].item()] += 1

				current_output_path = f"{output_path}/{class_index_to_string(labels[idx].item())}/{str(sample_count)}-Pred-{class_index_to_string(model_class_pred(full_model, image.to(device), device))}"

				if mode == "composite" or mode == "singlar":
					generate_concept_LRP_maps(models, rules, image.to(device), current_output_path, n_concepts, device, concept_index_to_string)
				elif mode == "IG":
					generate_concept_IG_maps(models[0], image.to(device), current_output_path, n_concepts, device=device, concept_index_to_string=concept_index_to_string)
				else:
					pass

				# Write true concepts to file
				with open(f'{current_output_path}/true-concepts.txt', mode='wt', encoding='utf-8') as f:
					f.write("True concepts for sample\n")
					f.write("========================\n")
					for idy, concept in enumerate(concepts[idx]):
						f.write(f"{idy}: {concept_index_to_string(idy)}, concept value: {concept.item()}\n")

				# save input image
				AX.clear()
				AX.set_axis_off()
				AX.imshow(orig_image[idx].squeeze().permute(1, 2, 0))
				FIG.savefig(f'{current_output_path}/input.png', bbox_inches='tight', pad_inches = 0)

				sample_count += 1


def generate_concept_LRP_maps(models, rules, image, save_dir, n_concept=112, device="cpu", concept_index_to_string=None):
	"""
	given a data sample, generate saliency maps with LRP for a given set of rule(s)
	and model parts(s)
	"""
	image = image.unsqueeze(0)
	image.requires_grad_(True)
	image.grad = None  # Reset gradient

	if not os.path.exists(save_dir):
		try:
			os.makedirs(save_dir)
		except OSError:
			print(f"Creation of the directory {save_dir} failed")
		else:
			pass

	# Loop over models in list (used for composite rule)
	# if len(models) == 1 then a single rule will be used, else composite rules
	input = None
	for idx, model in enumerate(models):
		# If image has not been passed through a model yet
		if input is None:
			input = image
		pred_concepts = model.forward(input, explain=True, rule=rules[idx])
		input = pred_concepts

	pred_concepts_readable = torch.nn.Sigmoid()(pred_concepts)

	pred_list = []
	for idx, i in enumerate(pred_concepts_readable[0]):
		pred_list.append((idx, i.item()))

	pred_concepts_readable = sorted(pred_list, key=lambda x: -x[1])

	# Write concept predictions to file and save
	with open(f'{save_dir}/concept-pred.txt', mode='wt', encoding='utf-8') as f:
		f.write("Predicted concepts (with Sigmoid)\n")
		f.write("=================================\n")
		for item in pred_concepts_readable:
			f.write(f"{item[0]}: {concept_index_to_string(item[0])}, Sig value: {item[1]}\n")

	# get saliecy maps for each concept
	for i in range(n_concept):
		filter_out = torch.zeros_like(pred_concepts)
		filter_out[:,i] += 1

		# Get the gradient of each input
		image_gradient = torch.autograd.grad(
			pred_concepts,
			image,
			grad_outputs=filter_out,
			retain_graph=True)[0]

		attr = heatmap(image_gradient, cmap_name='seismic')

		if concept_index_to_string != None:
			concept_name = concept_index_to_string(i)
		else:
			concept_name = ""

		AX.clear()
		AX.set_axis_off()
		AX.imshow(attr.squeeze(), cmap='seismic')
		FIG.savefig(f'{save_dir}/{i}-{concept_name}.png', bbox_inches='tight', pad_inches = 0)


def generate_concept_IG_maps(model, image_in, save_dir, n_concept=112, device="cpu", concept_index_to_string=None):
	"""
	given a data sample, generate saliency maps with IG
	"""
	# get predicted concets and save to txt file
	image = image_in.unsqueeze(0)

	if not os.path.exists(save_dir):
		try:
			os.makedirs(save_dir)
		except OSError:
			print(f"Creation of the directory {save_dir} failed")
		else:
			pass

	pred_concepts = model(image)

	pred_concepts_readable = torch.nn.Sigmoid()(pred_concepts)

	pred_list = []
	for idx, i in enumerate(pred_concepts_readable[0]):
		pred_list.append((idx, i.item()))

	pred_concepts_readable = sorted(pred_list, key=lambda x: -x[1])

	# Write concept predictions to file and save
	with open(f'{save_dir}/concept-pred.txt', mode='wt', encoding='utf-8') as f:
		f.write("Predicted concepts (with Sigmoid)\n")
		f.write("=================================\n")
		for item in pred_concepts_readable:
			f.write(f"{item[0]}: {concept_index_to_string(item[0])}, Sig value: {item[1]}\n")

	ig = IntegratedGradients(model)
	nt = NoiseTunnel(ig)

	image = image_in.unsqueeze(0)
	image.requires_grad_(True)
	image.grad = None  # Reset gradient
	baselines = image * 0


	#model.zero_grad()

	# get saliecy maps for each concept
	for i in range(n_concept):

		attributions_ig = nt.attribute(
			image,
			target=i,
			baselines=baselines,
			nt_type='smoothgrad_sq',
			nt_samples=25,
			internal_batch_size=25,
			stdevs=0.2
		)

		attr = heatmap(attributions_ig, cmap_name='seismic')

		if concept_index_to_string != None:
			concept_name = concept_index_to_string(i)
		else:
			concept_name = ""

		AX.clear()
		AX.set_axis_off()
		AX.imshow(attr.squeeze(), cmap='seismic')
		FIG.savefig(f'{save_dir}/{i}-{concept_name}.png', bbox_inches='tight', pad_inches = 0)

		print("3:", datetime.now() - startTime)

# C to Y generation

# concept importance


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
		'--use_torchexplain',
		action='store_true',
		help='use the package and convert model to work with torchexplain',
		default=True
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
		default="composite",
		choices=["composite", "singlar", "IG", "CtoY", "conceptImportance"],
		help='mode to run script'
	)
	parser.add_argument(
		'--samples_per_class',
		type=int,
		help='Max number of samples to generate results for per class in the dataset',
		default=10
	)
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		print("Device:", device, torch.cuda.get_device_name(0))
	else:
		print("Device:", device)

	# load model and dataset
	test_loader = load_data("./dataset/CUB/dataset_splits/CBM_dataset_split/test.pkl", 16, image_dir='dataset/CUB/data/images', return_orig=True)

	class_index_to_string = IndexToString("./dataset/CUB/data/classes.txt", classes=True)
	concept_index_to_string = IndexToString("./dataset/CUB/dataset_splits/CBM_dataset_split/attributes.txt")

	XtoCtoY_model = ModelXtoCtoY(n_class_attr=args.n_class_attr, num_classes=args.n_classes, n_attributes=args.n_concepts, expand_dim=args.expand_dim, use_relu=args.use_relu, use_sigmoid=args.use_sigmoid, train=args.use_torchexplain)
	XtoCtoY_model.load_state_dict(torch.load(args.model))
	XtoCtoY_model.to(device)
	XtoCtoY_model.eval()

	output_path = f"{args.output}/{args.mode}"

	# create list to count number of samples have results generated for per class
	sample_counter = []
	for i in range(args.n_classes):
		sample_counter.append(0)

	if args.mode == "composite":
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

		enumerate_data(model_parts, test_loader, output_path, class_index_to_string, concept_index_to_string, args.n_concepts, device, XtoCtoY_model, rules=args.rules, mode=args.mode, sample_counter=sample_counter)
	elif args.mode == "singlar":
		XtoC_model = convert(XtoCtoY_model.first_model)
		XtoC_model.to(device)
		XtoC_model.eval()
		print("model loaded")

		enumerate_data([XtoC_model], test_loader, output_path, class_index_to_string, concept_index_to_string, args.n_concepts, device, XtoCtoY_model, rules=args.rules, mode=args.mode, sample_counter=sample_counter)
	elif args.mode == "IG":
		XtoC_model = XtoCtoY_model.first_model
		XtoC_model.to(device)
		XtoC_model.eval()
		print("model loaded")

		enumerate_data([XtoC_model], test_loader, output_path, class_index_to_string, concept_index_to_string, args.n_concepts, device, XtoCtoY_model, mode=args.mode, sample_counter=sample_counter)
	elif args.mode == "CtoY":
		pass
	elif args.mode == "conceptImportance":
		pass
	else:
		print("mode not recognised")

	#assert len(model_parts) == len(args.composite_rules), "Number of model parts must match the number of rules"

	#enumerate_data(model_parts, args.composite_rules, test_loader, args.output, args.n_concepts, device)
	#print("Done composite LRP rules")
