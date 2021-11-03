import os
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class VGG_FeatureExtractor(nn.Module):
    """ FeatureExtractor of CRNN (https://arxiv.org/pdf/1507.05717.pdf) """

    def __init__(self, input_channel, output_channel=512):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), 
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x50

            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), 
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x8x25

            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), 
            nn.ReLU(True),  # 256x8x25

            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), 
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25

            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), 
            nn.ReLU(True),  # 512x4x25

            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), 
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25

            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), 
            nn.ReLU(True))  # 512x1x24

    def forward(self, input):
        return self.ConvNet(input)


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        # self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True, dropout=0.5)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        # self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        b, T, h = recurrent.size()
        recurrent = recurrent.reshape(b * T, h)
        output = self.linear(recurrent)  # batch_size x T x output_size
        output = output.view(b, T, -1)
        return output


class Model(nn.Module):
    def __init__(self, imgSize, backbone, inputChannel, outputChannel, rnnHiddenSize, numChars):

        super(Model, self).__init__()

        # Feature Extractor
        if backbone == "vgg":
            self.featureExtractor = VGG_FeatureExtractor(inputChannel, outputChannel)

        elif backbone == "resnet18":
            outputChannel = 256
            self.featureExtractor = nn.Sequential(
                *list(models.resnet18(pretrained=True).children())[:-3],

                nn.Conv2d(256, outputChannel, kernel_size=(3,6), stride=1, padding=1),
                nn.BatchNorm2d(outputChannel),
                nn.ReLU(inplace=True)
                )
        elif backbone == "resnet34":
            outputChannel = 256
            self.featureExtractor = nn.Sequential(
                *list(models.resnet34(pretrained=True).children())[:-3],

                nn.Conv2d(256, outputChannel, kernel_size=(3,6), stride=1, padding=1),
                nn.BatchNorm2d(outputChannel),
                nn.ReLU(inplace=True)
                )
            
        elif backbone == "resnet50":
            outputChannel = 1024
            self.featureExtractor = nn.Sequential(
                *list(models.resnet50(pretrained=True).children())[:-3],

                nn.Conv2d(1024, outputChannel, kernel_size=(3,6), stride=1, padding=1),
                nn.BatchNorm2d(outputChannel),
                nn.ReLU(inplace=True)
                )

        if (backbone[:6] == "resnet") and (inputChannel == 1):
            self.featureExtractor[0] = nn.Conv2d(inputChannel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        b, c, h, w = self.featureExtractor(torch.Tensor(1, inputChannel, imgSize[0], imgSize[1])).shape


        # Sequecnce Modeling
        # self.SequenceModel = nn.Sequential(
        #     BidirectionalLSTM(h * c, rnnHiddenSize, rnnHiddenSize),
        #     BidirectionalLSTM(rnnHiddenSize, rnnHiddenSize, rnnHiddenSize),
        #     nn.Linear(rnnHiddenSize, numChars)
        #     )

        self.SequenceModel = nn.Sequential(
            nn.Linear(h * c, rnnHiddenSize),
            BidirectionalLSTM(rnnHiddenSize, rnnHiddenSize, rnnHiddenSize),
            BidirectionalLSTM(rnnHiddenSize, rnnHiddenSize, rnnHiddenSize),
            nn.Linear(rnnHiddenSize, numChars)
            )


    def forward(self, batch):
        batch = self.featureExtractor(batch)

        batch = batch.permute(0, 3, 1, 2)

        B = batch.size(0)
        T = batch.size(1)
        C = batch.size(2)
        H = batch.size(3)

        batch = batch.view(B, T, C * H)

        batch = self.SequenceModel(batch)

        batch = batch.permute(1, 0, 2)

        batch = F.log_softmax(batch, 2)

        return batch

def weights_init(m):
    classname = m.__class__.__name__
    if type(m) in [nn.Linear, nn.Conv2d, nn.Conv1d]:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

if __name__ == "__main__":
    os.environ['TORCH_HOME'] = os.path.sep.join(["E:", "Models", "cache", "pytorch"])

    # model = VGG_FeatureExtractor(input_channel=3, output_channel=256)
    # a = torch.Tensor(1, 3, 32, 100)
    # output = model(a).permute(0, 3, 1, 2).squeeze(3)
    # print(output.shape)

    inputChannel = 1
    model = Model((50, 200), "resnet50", inputChannel, 256, 256, 37)
    output = model(torch.Tensor(1, inputChannel, 50, 200))
    print(output.shape)

    torch.save(model.state_dict(), "model.pth")