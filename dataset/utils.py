import os
import torch
import torchvision.transforms.functional as F
import numbers


class IndexToString(object):
	"""Convert index to the corrisponding string
	Args:
		file_path (string): path to txt file containing class ids (starting from 1) and strings
		classes (boot): True is the input file contains sample classes
	"""

	def __init__(self, file_path, classes=False):
		self.string_dict = {}
		with open(file_path) as f:
			lines = f.readlines()
			for i in lines:
				i = i.split(" ")
				if classes:
					String_value = (i[1][:-1]).split(".")[1]
				else:
					String_value = i[1][:-1]
				self.string_dict[int(i[0]) - 1] = String_value  # ids start from 1 but indexes start from 0

	def __call__(self, index):
		"""
		Args:
			index (int): index of class
		Returns:
			String: String name
		"""
		return self.string_dict[index]


def conceptToBirstPart(index):
	"""return list of bird parts (0 indexed) given a concept (0 indexed).

	This function will handle conversion between 1 and 0 indexing

	Args:
		index (int): index of concept
	Returns:
		list of bird part ids. If list is empty the concept does not translate to
		a bird part, if list has len > 1, the concept translates to multiple bird parts
	"""

	bird_parts = {
		1: [2],
		2: [2],
		3: [2],
		4: [2],
		5: [9, 13],
		6: [9, 13],
		7: [9, 13],
		8: [9, 13],
		9: [9, 13],
		10: [9, 13],
		11: [],
		12: [],
		13: [],
		14: [],
		15: [],
		16: [],
		17: [],
		18: [],
		19: [],
		20: [],
		21: [],
		22: [],
		23: [],
		24: [4],
		25: [4],
		26: [1],
		27: [1],
		28: [1],
		29: [1],
		30: [1],
		31: [1],
		32: [14],
		33: [14],
		34: [14],
		35: [14],
		36: [14],
		37: [14],
		38: [],
		39: [],
		40: [4],
		41: [4],
		42: [4],
		43: [4],
		44: [4],
		45: [4],
		46: [15],
		47: [15],
		48: [15],
		49: [15],
		50: [15],
		51: [7, 11],
		52: [7, 11],
		53: [2],
		54: [2],
		55: [6],
		56: [6],
		57: [6],
		58: [6],
		59: [6],
		60: [14],
		61: [14],
		62: [14],
		63: [14],
		64: [14],
		65: [10],
		66: [10],
		67: [10],
		68: [10],
		69: [10],
		70: [10],
		71: [3],
		72: [3],
		73: [3],
		74: [3],
		75: [3],
		76: [3],
		77: [3],
		78: [9, 13],
		79: [],
		80: [],
		81: [],
		82: [],
		83: [],
		84: [],
		85: [1],
		86: [1],
		87: [1],
		88: [14],
		89: [14],
		90: [14],
		91: [],
		92: [],
		93: [],
		94: [],
		95: [],
		96: [],
		97: [8, 12],
		98: [8, 12],
		99: [8, 12],
		100: [2],
		101: [2],
		102: [2],
		103: [2],
		104: [5],
		105: [5],
		106: [5],
		107: [5],
		108: [5],
		109: [5],
		110: [9, 13],
		111: [9, 13],
		112: [9, 13],
	}

	return [n - 1 for n in bird_parts[index + 1]]


# https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomHorizontalFlip
# modified to return bool == True if image was flipped
class RandomHorizontalFlip(torch.nn.Module):
	"""Horizontally flip the given image randomly with a given probability.
	If the image is torch Tensor, it is expected
	to have [..., H, W] shape, where ... means an arbitrary number of leading
	dimensions
	Args:
		p (float): probability of the image being flipped. Default value is 0.5
	"""

	def __init__(self, p=0.5):
		super().__init__()
		self.p = p

	def forward(self, img):
		"""
		Args:
			img (PIL Image or Tensor): Image to be flipped.
		Returns:
			PIL Image or Tensor: Randomly flipped image.
			Bool: True if image flipped, else False
		"""
		if torch.rand(1) < self.p:
			return F.hflip(img), True
		return img, False

	def __repr__(self):
		return f"{self.__class__.__name__}(p={self.p})"


# https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#CenterCrop
# modified to return coordinates of cropped input
class CenterCrop(torch.nn.Module):
	"""Crops the given image at the center.
	If the image is torch Tensor, it is expected
	to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
	If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.
	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
	"""

	def __init__(self, size):
		super().__init__()
		self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

	def forward(self, img):
		"""
		Args:
			img (PIL Image or Tensor): Image to be cropped.
		Returns:
			PIL Image or Tensor: Cropped image.
			List: [x1, y1, x2, y2] of cropped coordinates on input
		"""
		return center_crop(img, self.size)

	def __repr__(self):
		return f"{self.__class__.__name__}(size={self.size})"


# https://pytorch.org/vision/stable/_modules/torchvision/transforms/functional.html#center_crop
# modified to return coordinates of cropped input
def center_crop(img, output_size):
	"""Crops the given image at the center.
	If the image is torch Tensor, it is expected
	to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
	If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.
	Args:
		img (PIL Image or Tensor): Image to be cropped.
		output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
			it is used for both directions.
	Returns:
		PIL Image or Tensor: Cropped image.
		List: [x1, y1, x2, y2] of cropped coordinates on input
	"""
	if isinstance(output_size, numbers.Number):
		output_size = (int(output_size), int(output_size))
	elif isinstance(output_size, (tuple, list)) and len(output_size) == 1:
		output_size = (output_size[0], output_size[0])

	image_width, image_height = F.get_image_size(img)
	crop_height, crop_width = output_size

	if crop_width > image_width or crop_height > image_height:
		padding_ltrb = [
			(crop_width - image_width) // 2 if crop_width > image_width else 0,
			(crop_height - image_height) // 2 if crop_height > image_height else 0,
			(crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
			(crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
		]
		img = F.pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
		image_width, image_height = F.get_image_size(img)
		if crop_width == image_width and crop_height == image_height:
			return img, [0, 0, image_width, image_height]

	crop_top = int(round((image_height - crop_height) / 2.0))
	crop_left = int(round((image_width - crop_width) / 2.0))
	return F.crop(img, crop_top, crop_left, crop_height, crop_width), [crop_top, crop_left, crop_height, crop_width]


# https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#CenterCrop
def _setup_size(size, error_msg):
	if isinstance(size, numbers.Number):
		return int(size), int(size)

	if isinstance(size, Sequence) and len(size) == 1:
		return size[0], size[0]

	if len(size) != 2:
		raise ValueError(error_msg)

	return size
