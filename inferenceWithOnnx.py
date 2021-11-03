import cv2
import numpy as np

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

def process(image, imgChannel, model_path):

	if imgChannel == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
		image /= 255.0
		image -= [0.485, 0.456, 0.406]
		image /= [0.229, 0.224, 0.225]
	elif imgChannel == 1:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
		image -= 127.5
		image /= 127.5
	if "resnet" in model_path:
		blob = cv2.dnn.blobFromImage(image, size=(200, 50))
	elif "vgg" in model_path:
		blob = cv2.dnn.blobFromImage(image, size=(100, 32))
	else:
		blob = cv2.dnn.blobFromImage(image, size=(100, 32))

	return blob

if __name__ == "__main__":
	model_path = r"E:\Projects2\TextRecognition-Pytorch\check_points\vgg_256_1.onnx"
	imagePath = r"E:\Projects2\AnswerEvaluation\TestingImages\2.jpg"

	recognizer = cv2.dnn.readNetFromONNX(model_path)
	imgChannel = int(model_path.split(".onnx")[0][-1])
	image = cv2.imread(imagePath)

	blob = process(image, imgChannel, model_path)

	recognizer.setInput(blob)
	result = recognizer.forward()
	wordRecognized = decodeText(result)

	print(wordRecognized)