import torch
from torch.autograd import Variable
from PIL import Image
from os import listdir
from os.path import isfile, join
from collections import deque
import torchvision.models as models
import torchvision.transforms as transforms
import zipfile


# Load the pretrained model
model = models.resnet18(pretrained=True)
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

model.eval()
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(512)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data.reshape(o.data.size(1)))
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return embeddings

def get_maps(image_path):
    zip = zipfile.ZipFile(image_path)
    zlist = zip.namelist()
	#img_files = [f for f in listdir(image_path) if isfile(join(image_path, f))]
	im_vectors = deque()
	for i in range(len(zlist)):
        img_file = zip.open(zlist[i])
		im_vector = get_vector(img_file)
		im_vectors.append(im_vector)
	return im_vectors
