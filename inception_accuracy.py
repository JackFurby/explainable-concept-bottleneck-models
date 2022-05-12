from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset.dataset import load_data
from dataset.utils import *
import matplotlib.pyplot as plt

from models.inception_v3 import *
from CUB.models import End2EndModel

N_CONCEPT = 112


def generateClassMatrix(dataloader, model, device="cpu"):
	nb_classes = 200
	model.eval()
	confusion_matrix = torch.zeros(nb_classes, nb_classes, dtype=torch.int32)

	with torch.no_grad():
		for i, (images, labels, concepts) in enumerate(dataloader):
			images, labels = images.to(device), labels.to(device)
			predictions = model(images)
			_, preds = torch.max(predictions[0], 1)
			for t, p in zip(labels.view(-1), preds.view(-1)):
				confusion_matrix[t.item(), p.item()] += 1

	return confusion_matrix


def print_model_accuracy(first_model_path, second_model_path, use_sigmoid, device):

	use_relu = False
	use_sigmoid = use_sigmoid
	freeze = False
	expand_dim = 0
	pretrained = False
	train = True
	n_class_attr = 2

	if (first_model_path != None) and (second_model_path == None):
		XtoCtoY_model = torch.load(first_model_path)
		XtoCtoY_model.to(device)
		XtoCtoY_model.eval()
	else:
		XtoC_model = torch.load(first_model_path)
		CtoY_model = torch.load(first_model_path)

		XtoCtoY_model = End2EndModel(XtoC_model, CtoY_model, use_relu, use_sigmoid, n_class_attr)
		CtoY_model.to(device)
		CtoY_model.eval()
	print("model loaded")

	test_split = pickle.load(open("./dataset/CUB/CBM_dataset_split_train_test_val/test.pkl", "rb"))
	test_loader = load_data("./dataset/CUB/CBM_dataset_split_train_test_val/test.pkl", 16, image_dir='dataset/CUB/data/images')

	class_index_to_string = IndexToString("./dataset/CUB/data/classes.txt", classes=True)
	concept_index_to_string = IndexToString("./dataset/CUB/CBM_dataset_split_train_test_val/attributes.txt")

	IMG_SIZE =  299
	training_transform = transforms.Compose([
		transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
		transforms.RandomResizedCrop(IMG_SIZE),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(), #implicitly divides by 255
		transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
		#transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
		]
	)

	other_transform = transforms.Compose([
		transforms.CenterCrop(IMG_SIZE),
		transforms.ToTensor(), #implicitly divides by 255
		transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
		#transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
		]
	)

	matrix = generateClassMatrix(test_loader, XtoCtoY_model, device)

	print("Test accuracy")
	print("=============")
	print(str(torch.mean(matrix.diag()/matrix.sum(1)).item() * 100) + "%")
	print("")
	print("Per class accuracy")
	print("==================")
	for i, acc in enumerate(matrix.diag()/matrix.sum(1)):
		print(class_index_to_string(i) + ":", str(acc.item() * 100) + "%")
	print("")
	print("Confusion materix")
	print("=================")
	print(matrix)

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Train and test a Concept Bottleneck Model')
	parser.add_argument(
		'--first_model_path',
		type=str,
		help='XtoC or joint model path'
	)
	parser.add_argument(
		'--second_model_path',
		type=str,
		help='CtoY model path',
		default=None
	)
	parser.add_argument(
		'--use_sigmoid',
		action='store_true',
		help='Whether to include sigmoid activation before using attributes to predict Y. For end2end & bottleneck model'
	)
	args = parser.parse_args()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		print("Device:", device, torch.cuda.get_device_name(0))
	else:
		print("Device:", device)

	print_model_accuracy(args.first_model_path, args.second_model_path, args.use_sigmoid, device)
