import os
import random
import json, torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    
	def __init__(self, data_dir, image_fns, aug, jsonFilePath, imgSize=(50, 200), imgChannel=1):
		self.data_dir = data_dir
		self.image_fns = image_fns
		self.file = json.load(open(jsonFilePath))
		self.aug = aug

		if imgChannel == 3:
			self.augment_transform = transforms.Compose([
				transforms.Resize((imgSize[0], imgSize[1])),
				transforms.RandomApply(torch.nn.ModuleList([
					transforms.ColorJitter(brightness=0.1, hue=0.1, contrast=0.1),

					# transforms.ColorJitter(brightness=2, hue=.3),
					# transforms.RandomAutocontrast(),
					# transforms.RandomAffine(degrees=[-5.5, 5.5], shear=[-10.5, 10.5]),
					# transforms.RandomAdjustSharpness(sharpness_factor=0.02),

					transforms.RandomPerspective(distortion_scale=0.15, p=0.8),
					transforms.RandomAdjustSharpness(sharpness_factor=0.1, p=0.8),

					transforms.GaussianBlur(random.choice([1, 3, 5])),

					# transforms.RandomEqualize(p=0.5),

					transforms.Grayscale(3),
				]), p=0.8),
				transforms.ToTensor(),
				transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
			])
			self.transform = transforms.Compose([
				transforms.Resize((imgSize[0], imgSize[1])),
				transforms.ToTensor(),
				transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
			])
		elif imgChannel == 1:
			self.augment_transform = transforms.Compose([
				transforms.Resize((imgSize[0], imgSize[1])),
				transforms.RandomApply(torch.nn.ModuleList([
					transforms.ColorJitter(brightness=0.1, hue=0.1, contrast=0.1),

					# transforms.ColorJitter(brightness=2, hue=.3),
					# transforms.RandomAutocontrast(),
					# transforms.RandomAffine(degrees=[-5.5, 5.5], shear=[-10.5, 10.5]),
					# transforms.RandomAdjustSharpness(sharpness_factor=0.02),

					transforms.RandomPerspective(distortion_scale=0.15, p=0.8),
					transforms.RandomAdjustSharpness(sharpness_factor=0.1, p=0.8),

					transforms.GaussianBlur(random.choice([1, 3, 5]))
				]), p=0.8),
				transforms.Grayscale(),
				transforms.ToTensor(),
				transforms.Normalize((0.5,), (0.5,))
			])
			self.transform = transforms.Compose([
				transforms.Resize((imgSize[0], imgSize[1])),
				transforms.Grayscale(),
				transforms.ToTensor(),
				transforms.Normalize((0.5,), (0.5,))
			])
		else:
			raise ValueError("Input Channel can be 1 or 3")


	def __len__(self):
		return len(self.image_fns)

	def __getitem__(self, index):

		image_fn = self.image_fns[index]
		image_fp = os.path.join(self.data_dir, image_fn)
		image = Image.open(image_fp).convert('RGB')

		if self.aug:
			image = self.augment_transform(image)
			# image = self.transform(image)
		else:
			image = self.transform(image)

		text = self.file[image_fn].split(".")[0]
		return image, text

if __name__ == "__main__":

	import numpy as np
	import cv2, random
	data_path = "E:/dataset/TextRecognition/IIIT5K/data"
	jsonFilePath = data_path+".json"

	image_fns = os.listdir(data_path)
	imgSize = (200, 50)[::-1]
	imgChannel = 3
	trainset = TextDataset(data_path, image_fns, aug=True,
			jsonFilePath=jsonFilePath, imgSize=imgSize, imgChannel=imgChannel)
	batch_size = 32
	train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
	for image_batch, text_batch in train_loader:
		for image in image_batch:
			image = image.permute(1, 2, 0).numpy()
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			print(image.min(), image.max())
			cv2.imshow("image", image)
			cv2.waitKey(1000)
		exit()

	og_image = cv2.resize(cv2.imread(imagePath), imgSize)
	blob = cv2.dnn.blobFromImage(cv2.cvtColor(og_image, cv2.COLOR_BGR2GRAY), size=(200, 50), mean=127.5, scalefactor=1 / 127.5)

	image = Image.open(imagePath).convert('RGB')
	image = transform(augment_transform(image)).permute(1, 2, 0).numpy()
	print("og_image", og_image.max(), og_image.min(),"blob", blob.max(), blob.min(), blob.shape, "transform", image.max(), image.min(), image.shape)
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	cv2.imshow("image", image)
	cv2.imshow("og_image", og_image)
	cv2.waitKey(1000)

    # loader = TextDataset(data_path, image_fns, train=True, jsonFilePath=jsonFilePath)