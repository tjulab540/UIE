import argparse
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.autograd import Variable
from models import *
import torch.nn as nn
import torch.nn.functional as F
import torch

# Parameter initialization
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="water", help="name of the dataset")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
opt = parser.parse_args()


image_name = './test.png'
cuda = torch.cuda.is_available()
cuda = False
in_features = 256
out_features = 256
E_AB = Downsampling(opt.channels)
F_AB = Upsampling(in_features, out_features, opt.n_residual_blocks)
if cuda:
    E_AB = E_AB.cuda()
    F_AB = F_AB.cuda()

# Load pretrained models
E_AB.load_state_dict(torch.load("./models/E_AB_60.pth" ))
F_AB.load_state_dict(torch.load("./models/F_AB_60.pth" ))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Image transformations
transforms_ = [
    transforms.ToTensor(),
]
transform = transforms.Compose(transforms_)

def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image

image = Image.open(image_name)
if image.mode != "RGB":
    image = to_rgb(image)
image = transform(image).unsqueeze_(0)
E_AB.eval()  # Load model
F_AB.eval()
real_A = Variable(image.type(Tensor))
with torch.no_grad():
    feature_A = E_AB(real_A)
    fake_B = F_AB(feature_A)
save_image(fake_B, "./result.png" , normalize=True)