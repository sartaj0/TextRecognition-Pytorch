# TextRecognition-Pytorch 

## Onnx Model
From [here](https://drive.google.com/drive/folders/12S3AVgnK-hnjQvKHNyTiTIEIm_5MA7G3?usp=sharing) you can download onnx converted model and use this [file](https://github.com/sartaj0/TextRecognition-Pytorch/blob/main/inferenceWithOnnx.py) for inferencing with opencv. 

## Dependency 
- pytorch 1.8.1, CUDA 10.2 <br> `pip3 install torch==1.8.1+cu102 torchvision==0.9.1+cu102` 
- Requirements: Pillow, opencv-python, tqdm, matplotlib, nltk <br> `pip3 install Pillow opencv-python tqdm matplotlib nltk` 


## Dataset 
Download dataset from from [here](https://drive.google.com/drive/folders/1SGNmiD6FvZFS3Qjk5DILIQFTFi_vSx0g?usp=sharing)
data.zip contain below data. <br> 
[IIIT5k](http://cvit.iiit.ac.in/projects/SceneTextUnderstanding/IIIT5K.html)[1] [ic17](https://rrc.cvc.uab.es/?ch=8)[2] [ic03](http://www.iapr-tc11.org/mediawiki/index.php/ICDAR_2003_Robust_Reading_Competitions)[3] [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)[4]


## Training and Inference 

#### Dataset Format 
```
folder
├── data.json
└── data
    ├── word_1.png
    ├── word_2.png
    ├── word_3.png
```

At this time, `data.json` should be `{imagename: label, }` <br> 
For example `{word_1: label, ...}` <br>
#### 1. Train 

Set The Paramters 
```
from train import *
data_path = "E:/dataset/TextRecognition/MixedAll/data"
jsonFilePath = data_path+".json"

model_save_directory = "check_points"
batch_size = 32
rnn_hidden_size = 256
cnn_output_channel = 512
num_epochs = 50
model_backbone = "resnet18"
imgSize = (50, 200)
# imgSize = (32, 100)
imgChannel = 1
lr = 0.000087
train(imgSize, imgChannel, data_path, jsonFilePath, model_backbone, model_save_directory, num_epochs, cnn_output_channel, rnn_hidden_size, batch_size, lr)
```
<br>

#### 2. Validation 
From training data it will use 20% images for validation <br>


#### 3. Inference 
Once the training is completed it will use the best model for onnx conversion
you can use [inferenceWithOnnx.py](https://github.com/sartaj0/TextRecognition-Pytorch/blob/main/inferenceWithOnnx.py) for inferencing with opencv. 

![](https://github.com/sartaj0/GIfs/blob/main/2.jpg) <br> 
output: `develop` <br> 

<img src="https://github.com/sartaj0/GIfs/blob/main/q.jpg" width="500" title="failure cases"> <br>
output <br>

***
dassmate date page the water cucle is thie palith that al watey follows as at moves asound eanth in dpfestent ptat liouid water hound in oceans o rivers akee and even under raround solid iee ie found is alaciers onow and at the nortr and south sboles water vadoy a aas do found in farlhs atmos here the sun heat causce alaceers and snow se te melt into liouid water this watey sanpe intt oceans lakes and streams watey foom ce melbino snow and pe rlkd aoes iito e tthe 50ic tiere at suddlie water st tos folants and the around waa ter that ce we drink worrmi uaterr na bouy siscs up throuan easltis atmuss ene as the watey vaour viscs hiaher and hiahea the cool air ot the itmos causes the waler vapoly to ttuyn back snte lrauid water creatina clouds this brocess i5 called condenstin uhen a cloud be fuu a lrouid usater it falle toom the skn as rain
***

@InProceedings{MishraBMVC12,
  author    = "Mishra, A. and Alahari, K. and Jawahar, C.~V.",
  title     = "Scene Text Recognition using Higher Order Language Priors",
  booktitle = "BMVC",
  year      = "2012",
}

## References

- [zihaomu](https://github.com/zihaomu/deep-text-recognition-benchmark) 
- [gokul karthik](https://www.kaggle.com/gokulkarthik/captcha-text-recognition-using-crnn-in-pytorch) 
- [nanonets](https://nanonets.com/blog/deep-learning-ocr/) 