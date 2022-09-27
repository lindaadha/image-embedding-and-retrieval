from tkinter import image_names
from flask import Flask, jsonify, request, flash, redirect
from io import BytesIO
import torchvision.models as models
import torchvision.transforms as transforms
from torch import nn
from sklearn.metrics.pairwise import cosine_similarity
import torch, math, json, base64, cv2
import numpy as np
from PIL import Image

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

def prediction(dict, feature):
    dict_img= []
    dict_angle = []
    list = []
    with open(dict, "r") as outfile:
        dict_obj =json.load(outfile)

    best = 50
    for key in dict_obj:
        value = dict_obj[key]
        value = np.array(value)
        angle = calculate_angle(feature, value)
        # print(angle)
        if angle < best:
            # best = angle
            best_id = key
            dict_angle.append(angle)
            dict_img.append(best_id)

    return dict_img, dict_angle

def sort_angle(dict_img, dict_angle):
    index_sort = np.argsort(dict_angle)
    dict_angle.sort()

    best_predict_index = index_sort[:10]
    best_predict_name = np.array(dict_img)[best_predict_index]

    best_angle = []
    best_base64 = []
    for i, predict_name in enumerate (best_predict_name):
        best_angle.append(dict_angle[i])
        img = Image.open(predict_name)
        img_base64 = convert_base64(img)
        best_base64.append(img_base64)

    return best_base64, best_angle, best_predict_name

def convert_base64 (img):
    # print (img_name[0])
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()
    im_b64 = base64.b64encode(im_bytes)
    return im_b64

def get_retrieval(image):
    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    img = Image.open(image)
    x = transform(img)
    x = x.unsqueeze(0)
    model = load_model(x)
    dict_img, dict_angle = prediction('object3.json', model)
    img64, angles, filename = sort_angle(dict_img, dict_angle)

    return img64, angles, filename



if __name__ == "__main__":
    # img_path = 'data'
    img_name = '24antelope.jpg'
    # model = load_model(img_name)
    # predict = prediction('object3.json', model)
    # img_list = convert_base64(predict)
    img64, angles, filename = get_retrieval(img_name)

    for data in zip(img64, angles, filename):
        print(data)
    
    # print(angles, filename)