import torch
import numpy as np
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
import math
from sklearn.metrics.pairwise import cosine_similarity
import os, glob
import cv2
import json

def load_model(x):
    model = models.resnet18(pretrained=True)
    model.eval()
    # print(model)
    model.fc = nn.Sequential()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = x.to(device)
    model.to(device)
    with torch.no_grad():
        output = model(x)
    return output

def calculate_angle(emb1, emb2):
    #print(np.shape(emb1), np.shape(emb2))
    cos_sim=cosine_similarity(emb1.reshape(1,-1),emb2.reshape(1,-1))
    angle = math.acos(cos_sim[0][0])
    angle = math.degrees(angle)
    return angle

transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
    dict_object = {}
    dict_result = []
    for name in glob.glob('data2/*/*.jpg'):
        # print(name)
        img = Image.open(name)
        x = transform(img)
        x = x.unsqueeze(0)
        # print(x.shape)
        best_result = load_model(x)

        # dict_object.append(name)
        # dict_result.append(best_result)

        # print(dict_object)

        dict_object[name] = best_result.tolist()

with open("object3.json", "a+") as outfile:
    json.dump(dict_object, outfile)
            