import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import models


from modules.dataloader import TextDataset
from modules.model import *

import json
from collections import Counter

def encode_text_batch(text_batch, char2idx):
    
    text_batch_targets_lens = [len(text) for text in text_batch]
    text_batch_targets_lens = torch.IntTensor(text_batch_targets_lens)
    
    text_batch_concat = "".join(text_batch)
    text_batch_targets = [char2idx[c] for c in text_batch_concat]
    text_batch_targets = torch.IntTensor(text_batch_targets)
    return text_batch_targets, text_batch_targets_lens

def compute_loss(criterion, text_batch, text_batch_logits, device, char2idx):
    """
    text_batch: list of strings of length equal to batch size
    text_batch_logits: Tensor of size([T, batch_size, num_classes])
    """
    # text_batch_logps = F.log_softmax(text_batch_logits, 2) # [T, batch_size, num_classes]  
    text_batch_logps = text_batch_logits
    text_batch_logps_lens = torch.full(size=(text_batch_logps.size(1),), 
                                       fill_value=text_batch_logps.size(0), 
                                       dtype=torch.int32).to(device) # [batch_size] 
    # print(text_batch_logps.shape)
    #print(text_batch_logps_lens) 
    text_batch_targets, text_batch_targets_lens = encode_text_batch(text_batch, char2idx)
    # print(text_batch_targets)
    # print(text_batch_targets_lens)
    loss = criterion(text_batch_logps, text_batch_targets, text_batch_logps_lens, text_batch_targets_lens)

    return loss


def save_loss_image(train_loss, val_loss, epoch, model_save_name, model_save_directory):

        fig = plt.figure()
        plt.plot([k for k in range(1, epoch + 1)], train_loss, label = "Training Loss")
        plt.plot([k for k in range(1, epoch + 1)], val_loss, label = "Validation Loss")
        plt.legend()
        plt.title(model_save_name)
        fig.canvas.draw()
        img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(model_save_directory, f"{model_save_name}_loss.jpg"), img)

        plt.close()

        # plt.savefig('loss.png')



def train(imgSize, imgChannel, data_path, jsonFilePath, model_backbone, model_save_directory, 
    num_epochs, cnn_output_channel, rnn_hidden_size, batch_size, lr):

    # data_path, model_save_directory
    image_fns = os.listdir(data_path)

    if not os.path.isdir(model_save_directory):
        os.mkdir(model_save_directory)

    '''
    model_save_directory = os.path.join(model_save_directory, str(rnn_hidden_size))
    if not os.path.isdir(model_save_directory):
        os.mkdir(model_save_directory)
    '''
    print("Number of Images:", len(image_fns))

    np.random.seed(2)
    np.random.shuffle(image_fns)
    split_size = int(len(image_fns) *  0.8)
    image_fns_train, image_fns_test = image_fns[:split_size], image_fns[split_size:]


    image_ns = "0123456789abcdefghijklmnopqrstuvwxyz"
    letters = sorted(list(set(list(image_ns))))
    print("Number of Letters:", len(letters), "Letters:", letters)


    vocabulary = ["-"] + letters
    idx2char = {k:v for k, v in enumerate(vocabulary, start=0)}
    char2idx = {v:k for k,v in enumerate(vocabulary, start=0)}
    num_chars = len(char2idx)
    print(char2idx)

    jsonFile = json.load(open(jsonFilePath))
    print(Counter(''.join(list(jsonFile.values()))))

    trainset = TextDataset(data_path, image_fns_train, aug=True,
     jsonFilePath=jsonFilePath, imgSize=imgSize, imgChannel=imgChannel)
    testset = TextDataset(data_path, image_fns_test, aug=False,
     jsonFilePath=jsonFilePath, imgSize=imgSize, imgChannel=imgChannel)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    print(f"Number of Training Sample: {len(train_loader)}, \nNumber of Testing Samples: {len(test_loader)}")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Model will run on:", device)

    weight_decay = 1e-3
    clip_norm = 5
    lr_scheduler_type = "ReduceLROnPlateau"


    # crnn = CRNN(num_chars, rnn_input_size=256, rnn_hidden_size=rnn_hidden_size, backbone=model_backbone)
    crnn = Model(imgSize, model_backbone, imgChannel, cnn_output_channel, rnn_hidden_size, num_chars)

    print(crnn)

    optimizer = optim.Adam(crnn.parameters(), lr=lr, weight_decay=weight_decay)
    # optimizer = optim.SGD(crnn.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)

    if lr_scheduler_type == "ReduceLROnPlateau":
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=5)
    elif lr_scheduler_type == "LambdaLR":
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.96 ** epoch)

    criterion = nn.CTCLoss(blank=0)
    crnn = crnn.to(device)

    epoch_losses = []
    num_updates_epochs = []

    train_each_epoch_loss = []
    val_each_epoch_loss = []

    model_save_name = f"{model_backbone}_{rnn_hidden_size}_{imgChannel}"
    PATH = os.path.join(model_save_directory, f"{model_save_name}.pth")

    min_val_loss = None
    for epoch in range(1, num_epochs+1):
        epoch_loss_list = [] 
        with tqdm(train_loader, leave=False, desc="Training") as tepoch:
            for image_batch, text_batch in tepoch:
                optimizer.zero_grad()
                text_batch_logits = crnn(image_batch.to(device))
                loss = compute_loss(criterion,text_batch, text_batch_logits, device, char2idx)
                iteration_loss = loss.item()

                tepoch.set_postfix(loss=iteration_loss)

                if np.isnan(iteration_loss) or np.isinf(iteration_loss):
                    continue
                epoch_loss_list.append(iteration_loss)
                loss.backward()
                nn.utils.clip_grad_norm_(crnn.parameters(), clip_norm)
                optimizer.step()

        epoch_loss = np.mean(epoch_loss_list)

        train_each_epoch_loss.append(epoch_loss)

        crnn.eval()
        with torch.no_grad():
            epoch_loss_list = [] 
            with tqdm(test_loader, leave=False, desc="Validation") as tepoch:
                for image_batch, text_batch in tepoch:
                    text_batch_logits = crnn(image_batch.to(device))
                    loss = compute_loss(criterion, text_batch, text_batch_logits, device, char2idx)

                    iteration_loss = loss.item()

                    epoch_loss_list.append(iteration_loss)

                    tepoch.set_postfix(loss=iteration_loss)

            val_loss = np.mean(epoch_loss_list)
            if (min_val_loss is None) or (min_val_loss > val_loss):
                min_val_loss = val_loss
                torch.save(crnn.state_dict(), PATH)

            time = datetime.now()
            print(f"Epoch:{epoch}\t Training Loss:{round(epoch_loss, 5)}\t Testing Loss:{round(val_loss, 5)}"+
                f"\t Learning Rate: {optimizer.param_groups[0]['lr']}\t {time.hour}:{time.minute}:{time.second}")

            val_each_epoch_loss.append(val_loss)

        crnn.train()

        save_loss_image(train_each_epoch_loss, val_each_epoch_loss, epoch, model_save_name, model_save_directory)

        if (np.isnan(epoch_loss) and np.isnan(val_loss)):
            break
            # crnn.load_state_dict(torch.load(PATH))
            # optimizer.param_groups[0]["lr"] = lr - ((lr - (lr / 100)) / num_epochs) * epoch

        if lr_scheduler_type == "LrDecay":
            optimizer.param_groups[0]["lr"] = lr - ((lr - (lr / 100)) / num_epochs) * epoch
        elif lr_scheduler_type == "ReduceLROnPlateau":
            lr_scheduler.step(val_loss)
        elif lr_scheduler_type == "LambdaLR":
            lr_scheduler.step()

    crnn.load_state_dict(torch.load(PATH))
    crnn.to("cpu")
    crnn.eval()

    dummy_input = torch.randn(1, imgChannel, imgSize[0], imgSize[1])
    torch.onnx.export(crnn, dummy_input,
        os.path.join(model_save_directory, f"{model_save_name}.onnx"), 
        verbose=True)



if __name__ == "__main__":
    data_path = "E:/dataset/TextRecognition/MixedAll/data"
    jsonFilePath = data_path+".json"
    os.environ['TORCH_HOME'] = os.path.sep.join(["E:", "Models", "cache", "pytorch"])


    model_save_directory = "check_points"
    batch_size = 32
    rnn_hidden_size = 256
    cnn_output_channel = 512
    num_epochs = 50
    model_backbone = "resnet50"
    imgSize = (50, 200)
    # imgSize = (32, 100)
    imgChannel = 3
    lr = 0.0001
    # lr = 0.000087
    # lr = 0.1


    train(imgSize, imgChannel, data_path, jsonFilePath, model_backbone, model_save_directory, 
        num_epochs, cnn_output_channel, rnn_hidden_size, batch_size, lr)