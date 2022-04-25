import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

class CNN_Encoder(nn.Module):
	def __init__(self, in_channels, d_model):
		super(CNN_Encoder, self).__init__()
		self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=(1,1), padding= (1,1))
		self.pool = nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
		self.fc = nn.Linear(64*2*2, d_model)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = self.pool(x)
		x = F.relu(self.conv2(x))
		x = self.pool(x)
		x = x.reshape(x.shape[0], -1)
		x = self.fc(x)

		return x