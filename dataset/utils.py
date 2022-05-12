import os


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
