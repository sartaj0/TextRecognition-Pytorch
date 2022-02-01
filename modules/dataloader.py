import os
import random
import json, torch
from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader


# class RandomChoice(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.t = random.choice(self.transforms)

#     def __call__(self, img):
#         return self.t(img)

def transformation(imgSize, imgChannel, aug=True, Normalizer=None):
	transform = [T.Resize((imgSize[0], imgSize[1])),
				T.ToTensor(),
				T.Normalize(mean=Normalizer[0], std=Normalizer[1])]
	if imgChannel == 1:
		transform.insert(1, T.Grayscale())

	if aug:
		transform.insert(1, 
			T.RandomApply(transforms=[
				T.RandomChoice([
					T.ColorJitter(brightness=0.1, hue=0.1, contrast=0.1),
					T.RandomAdjustSharpness(sharpness_factor=0.1, p=0.8),
					T.Grayscale(3),
					]),
				T.RandomChoice([
					T.RandomPerspective(distortion_scale=0.15, p=0.8),
					T.RandomAffine(degrees=(-5, 5), scale=(0.85, 0.95))
					])
				], p=0.8))

	transform = T.Compose(transform)
	print(transform)
	return transform



class TextDataset(Dataset):
    
	def __init__(self, data_dir, image_fns, aug, jsonFilePath, imgSize=(50, 200), imgChannel=1):
		self.data_dir = data_dir
		self.image_fns = image_fns
		self.file = json.load(open(jsonFilePath))
		self.aug = aug
		if imgChannel == 3:
			Normalizer = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
			Normalizer = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		elif imgChannel == 1:
			Normalizer = ((0.5,), (0.5,))
		self.transform = transformation(imgSize, imgChannel, self.aug, Normalizer)
		

	def __len__(self):
		return len(self.image_fns)

	def __getitem__(self, index):

		image_fn = self.image_fns[index]
		image_fp = os.path.join(self.data_dir, image_fn)
		image = Image.open(image_fp).convert('RGB')

		image = self.transform(image)

		text = self.file[image_fn].split(".")[0]
		return image, text

if __name__ == "__main__":

	import numpy as np
	import cv2, random
	# data_path = "E:/dataset/TextRecognition/IIIT5K/data"
	data_path = r"E:\dataset\ANPR\Final\data"
	jsonFilePath = data_path+".json"

	image_fns = os.listdir(data_path)
	imgSize = (200, 50)[::-1]
	imgChannel = 3
	trainset = TextDataset(data_path, image_fns, aug=True, jsonFilePath=jsonFilePath, imgSize=imgSize, imgChannel=imgChannel)
	batch_size = 32
	train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
	for image_batch, text_batch in train_loader:
		for image in image_batch:
			image = image.permute(1, 2, 0).numpy()
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			image *= 0.5
			image += 0.5
			image *= 255
			image = image.astype(np.uint8)
			print(image.min(), image.max(), image.shape)
			cv2.imshow("image", image)
			if cv2.waitKey(1000) == ord('q'):
				break
		exit()
