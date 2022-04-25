import torch
import torch.nn as nn
import torchvision.models as models

class PretrainedCNN(nn.Module):
	def __init__(self, embed_size, train_CNN = False):
		super(PretrainedCNN, self).__init__()
		self.train_CNN = train_CNN
		self.resnet18 = models.resnet18(pretrained=True)
		self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, embed_size)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.5)

	def forward(self, images):
		features = self.resnet18(images)

		for name, param in self.resnet18.named_parameters():
			if "fc.weight" in name or "fc.bias" in name:
				param.requires_grad = True 
			else:
				param.requires_grad = self.train_CNN

		return self.dropout(self.relu(features))