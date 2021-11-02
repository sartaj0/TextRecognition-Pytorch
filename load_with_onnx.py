import cv2
import os
import numpy as np
from nltk import edit_distance


def decodeText(scores):
	text = ""
	alphabet = "0123456789abcdefghijklmnopqrstuvwxyz"
	for i in range(scores.shape[0]):
		c = np.argmax(scores[i][0])
		if c != 0:
			text += alphabet[c - 1]
		else:
			text += '-'

	# adjacent same letters as well as background text must be removed to get the final output
	char_list = []
	for i in range(len(text)):
		if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
			char_list.append(text[i])
	return ''.join(char_list)


folder = r"E:/dataset/TextRecognition/test"
model_folder = "check_points"
model_folder_copy = model_folder


for model in os.listdir(model_folder):
	if not model.endswith(".onnx"):
		continue
	model_path = os.path.join(model_folder, model)
	recognizer = cv2.dnn.readNetFromONNX(model_path)

	imgChannel = int(model_path.split(".onnx")[0][-1])
	print(imgChannel)

	print(f"Using Model: {model}")
	for img in os.listdir(folder):
		image = cv2.imread(os.path.join(folder, img))

		if imgChannel == 3:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
			image /= 255.0
			image -= [0.485, 0.456, 0.406]
			image /= [0.229, 0.224, 0.225]
			blob = cv2.dnn.blobFromImage(image, size=(200, 50))

		elif imgChannel == 1:
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			blob = cv2.dnn.blobFromImage(image, size=(100, 32), mean=127.5, scalefactor=1 / 127.5)

		gtText = img.split(".")[0]

		# print(blob.max(), blob.min(), blob.shape)
		# exit()

		recognizer.setInput(blob)
		result = recognizer.forward()
		wordRecognized = decodeText(result)

		# Character Level Error Rate
		print(wordRecognized, "			", gtText, edit_distance(wordRecognized, gtText) / len(gtText))
		

	print("\n")

# Word Error Rate
print(edit_distance("myy nime iz kenneth", "my name is kenneth") / len("my name is kenneth".split(" ")))