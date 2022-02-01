import cv2
import os, time
import numpy as np
from inferenceWithOnnx import process, decodeText
from nltk import edit_distance

# folder = r"E:/dataset/TextRecognition/test"
folder = r"E:\ScorpiusSolution\ANPR_VehiclePassing\sample\malaysian_images"
model_folder = "check_points"
model_folder_copy = model_folder

os.system('cls')
start = time.time()
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

		blob = process(image, imgChannel, model_path)

		gtText = img.split(".")[0]

		recognizer.setInput(blob)
		result = recognizer.forward()
		wordRecognized = decodeText(result)

		# Character Level Error Rate
		print(wordRecognized, "			", gtText, edit_distance(wordRecognized, gtText) / len(gtText))
		

	print("\n")
print(time.time() - start)
# Word Error Rate
print(edit_distance("myy nime iz kenneth", "my name is kenneth") / len("my name is kenneth".split(" ")))
