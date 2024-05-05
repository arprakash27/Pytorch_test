# For tips on running notebooks in Google Colab, see
# https://pytorch.org/tutorials/beginner/colab

# get_ipython().run_line_magic('matplotlib', 'inline')
# %matplotlib inline


# import numpy as np
# import json
# import requests
import torch
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import time
import glob
import torchvision.transforms as transforms
# from torchvision import datasets, transforms


warnings.filterwarnings('ignore')
plt.show()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

resnet50 = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')

resnet50.eval().to(device)

imgs_ = glob.glob("Images/*.jp*g")

for img in imgs_:
    try:
        img = PIL.Image.open(img)
        print(img)
    except PIL.UnidentifiedImageError:
        print(img)


start = time.time()
batch = torch.cat([utils.prepare_input_from_uri(img) for img in imgs_]).to(device)


with torch.no_grad():
    output = torch.nn.functional.softmax(resnet50(batch), dim=1)

results = utils.pick_n_best(predictions=output, n=5)
end = time.time()

torch.save(resnet50.state_dict(), 'saved_model.pth')

for img, result in zip(imgs_, results):
    img = PIL.Image.open(img)
    # img.thumbnail((256, 256), Image.LANCZOS)
    plt.imshow(img)
    plt.text(0, 0, result)
    plt.show()
    # print(result)
print("Average Inference time per image:", (end-start)/6, "s")
