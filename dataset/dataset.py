import os
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader
import pickle

from dataset.utils import RandomHorizontalFlip as CustomRandomHorizontalFlip
from dataset.utils import CenterCrop as CustomCenterCrop

# General
BASE_DIR = ''
N_ATTRIBUTES = 312
N_CLASSES = 200


class CUBDataset(Dataset):
	"""
	Returns a compatible Torch Dataset object customized for the CUB dataset
	"""

	def __init__(
		self,
		pkl_file_path,
		dataset_root,
		transform=None,
		train=True,
		return_orig=False,
		return_path=False,
		return_locs=False,
		resol=299):
		"""
		Args:
			pkl_file_path (string): path to pkl file for dataset split from repo root
			dataset_root (string): path to dataset root from repo root
			transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
			train (bool): True if dataset contains training data else False
			return_orig (bool): If True then return the original image
			return_path (bool): If True then return the original image path (not used if return_orig is true)
			return_locs (bool): If True then return the original X and Y coordinates of bird parts visible
			resol (int): Size of resized image
		"""
		self.data = []
		self.train = train
		self.transform = transform
		self.return_orig = return_orig
		self.return_path = return_path
		self.return_locs = return_locs
		self.resol = resol

		dataset_split = pickle.load(open(pkl_file_path, 'rb'))

		for i in dataset_split:
			locs = i["locs"]
			locs.sort(key=lambda x:x['part_id'])
			sample = {
				"id": i["id"],
				"img_path": os.getcwd() + "/" + dataset_root + "/" + i["img_path"],
				"class_index": i["class_label"] - 1,  # class_no start from 1 and indexes start from 0
				#"concepts": [1 if x + 1 in i["attribute_no_list"] else 0 for x in range(N_ATTRIBUTES)]  # attribute_no_list starts from 1 and indexes start from 0
				"concepts": i["attribute_label"],
				"locs": locs

			}
			self.data.append(sample)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		img_data = self.data[idx]
		img_path = img_data['img_path']

		img = Image.open(img_path).convert('RGB')
		class_index = img_data['class_index']
		concepts = img_data['concepts']

		# Return bird part locations
		if self.return_locs:
			self.customCenterCrop = CustomCenterCrop(self.resol)
			locs = []

			for part in img_data['locs']:
				locs.append([part["x"], part["y"]])  # in order of id (first element has part_id == 0 etc.)
				#locs.append([part["x"], part["y"], part["visible"]])

			locs = np.array(locs, dtype='float32')

			img, cropCoordinates = self.customCenterCrop(img) # cropCoordinates = [crop_top, crop_left, crop_height, crop_width]

			# translate and scale points to coordinates after crop
			locs = locs - [cropCoordinates[1], cropCoordinates[0]]
			locs = locs * [self.resol / cropCoordinates[3], self.resol / cropCoordinates[2]]

			# set all points not in crop to 0
			locs[np.any(locs >= self.resol, axis=1)] = 0
			locs[np.any(locs <= 0, axis=1)] = 0

			# Can we move this to self.transform
			transform = transforms.Compose([
				transforms.ToTensor(), #implicitly divides by 255
				transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
				])

			altered_img = transforms.ToTensor()(img)
			altered_img = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])(altered_img)
			if self.return_orig:
				# hard coded transform without normalise (can I make this better?)
				img = transforms.ToTensor()(img)
				img = transforms.CenterCrop(self.resol)(img)
				return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), torch.as_tensor(locs, dtype=torch.float), torch.as_tensor(img)
			return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), torch.as_tensor(locs, dtype=torch.float)
		else:

			if self.transform is not None:
				altered_img = self.transform(img)
			else:
				altered_img = img

			if self.return_orig:
				# hard coded transform without normalise (can I make this better?)
				img = transforms.ToTensor()(img)
				img = transforms.CenterCrop(self.resol)(img)
				return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), torch.as_tensor(img)
			elif self.return_path:
				return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float), img_path
			else:
				return torch.as_tensor(altered_img), torch.as_tensor(class_index), torch.as_tensor(concepts, dtype=torch.float)


# https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
	"""Samples elements randomly from a given list of indices for imbalanced dataset
	Arguments:
		indices (list, optional): a list of indices
		num_samples (int, optional): number of samples to draw
	"""

	def __init__(self, dataset, indices=None):
		# if indices is not provided,
		# all elements in the dataset will be considered
		self.indices = list(range(len(dataset))) \
			if indices is None else indices

		# if num_samples is not provided,
		# draw `len(indices)` samples in each iteration
		self.num_samples = len(self.indices)

		# distribution of classes in the dataset
		label_to_count = {}
		for idx in self.indices:
			label = self._get_label(dataset, idx)
			if label in label_to_count:
				label_to_count[label] += 1
			else:
				label_to_count[label] = 1

		# weight for each sample
		weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
		self.weights = torch.DoubleTensor(weights)

	def _get_label(self, dataset, idx):  # Note: for single attribute dataset
		return dataset.data[idx]['concepts'][0]

	def __iter__(self):
		return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

	def __len__(self):
		return self.num_samples


# https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
def find_class_imbalance(pkl_file, multiple_attr=False, attr_idx=-1):
	"""
	Calculate class imbalance ratio for binary attribute labels stored in pkl_file
	If attr_idx >= 0, then only return ratio for the corresponding attribute id
	If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
	"""
	imbalance_ratio = []
	data = pickle.load(open(os.path.join(BASE_DIR, pkl_file), 'rb'))
	n = len(data)
	n_attr = len(data[0]['attribute_label'])
	if attr_idx >= 0:
		n_attr = 1
	if multiple_attr:
		n_ones = [0] * n_attr
		total = [n] * n_attr
	else:
		n_ones = [0]
		total = [n * n_attr]
	for d in data:
		labels = d['attribute_label']
		if multiple_attr:
			for i in range(n_attr):
				n_ones[i] += labels[i]
		else:
			if attr_idx >= 0:
				n_ones[0] += labels[attr_idx]
			else:
				n_ones[0] += sum(labels)
	for j in range(len(n_ones)):
		imbalance_ratio.append(total[j]/n_ones[j] - 1)
	if not multiple_attr: #e.g. [9.0] --> [9.0] * 312
		imbalance_ratio *= n_attr
	return imbalance_ratio


# https://github.com/yewsiang/ConceptBottleneck/blob/master/CUB/dataset.py
def load_data(pkl_path, batch_size, image_dir='.dataset/CUB/images', resampling=False, resol=299, return_orig=False, return_path=False, return_locs=False):
	"""
	Note: Inception needs (299,299,3) images with inputs scaled between -1 and 1
	Loads data with transformations applied, and upsample the minority class if there is class imbalance and weighted loss is not used
	NOTE: resampling is customized for first attribute only, so change sampler.py if necessary
	"""
	resized_resol = int(resol * 256/224)
	is_training = True if 'train.pkl' in pkl_path else False
	if is_training:
		transform = transforms.Compose([
			#transforms.Resize((resized_resol, resized_resol)),
			#transforms.RandomSizedCrop(resol),
			transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
			transforms.RandomResizedCrop(resol),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(), #implicitly divides by 255
			transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
			#transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
			])
	else:
		transform = transforms.Compose([
			#transforms.Resize((resized_resol, resized_resol)),
			transforms.CenterCrop(resol),
			transforms.ToTensor(), #implicitly divides by 255
			transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [2, 2, 2])
			#transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
			])

	dataset = CUBDataset(pkl_path, image_dir, transform, train=is_training, return_orig=return_orig, return_path=return_path, return_locs=return_locs)
	if is_training:
		drop_last = True
		shuffle = True
	else:
		drop_last = False
		shuffle = False
	if resampling:
		sampler = BatchSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size)
		loader = DataLoader(dataset, batch_sampler=sampler)
	else:
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
	return loader
