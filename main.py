from io import StringIO
from pathlib import Path
from numpy import result_type
import streamlit as st
import time
from detect import detect
import os
import sys
import argparse
from PIL import Image
import plotly.express as px

# Preprocessing of the input image
from torchvision import transforms
from torchvision import models
import torch
def preprocess_ImageNet(img):
    
    # define the preprocessor
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])


    # pass the input image through the preprocessor
    img_t = preprocess(img)
    
    # create a batch
    img_t = img_t[:3]
    batch_t = torch.unsqueeze(input=img_t, dim=0) # insert out image in batch
        
    return batch_t

def postprocess(output_data, topPred=1):
    # get class names
    with open('data/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
        
    # calculate human-readable value by softmax
    confidences = torch.nn.functional.softmax(output_data, dim=1)[0] * 100

    # find top-5 predicted classes
    _, indices = torch.sort(output_data, descending=True)
    i = 0
    
    # print the top classes predicted by the model
    keepPredClass = []
    keepPredConf = []
    
    while i<topPred:
        class_idx = indices[0][i]
        
        print( "Predicted class: {0}\t Confidence: {1:2.2f}%".format(
                                classes[class_idx], 
                                confidences[class_idx].item()))
        
        keepPredClass.append(classes[class_idx])
        keepPredConf.append(float(confidences[class_idx]))
        i += 1    
    
    return {'class': keepPredClass,
            'conf': keepPredConf}



def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
   
    return result
  

def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


if __name__ == '__main__':

    st.markdown("<h1 style='text-align: left; font:aharoni; font-weight:bold;color:white;font-size:35pt;'> Object Detection📸</h1>",unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; font:aharoni;color:white;font-size:13pt;'> 🌑Detect objects on blurry, partially visible and low illumination conditions🌑</h3>",unsafe_allow_html=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='weights/yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str,
                        default='data/images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.35, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true',
                        help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true',
                        help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    source = ("Image Detection📸", "Camera live Detection","Video Detection🎥")
    source_index = st.sidebar.selectbox("Select Object detection👇🏻", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "Upload Image🔽", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Detecting with 💕 ...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                imgGray = picture.convert('L')
                
                picture = imgGray.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
                st.write("")
                st.image(uploaded_file, caption='Original Picture') 
                
        else:
            is_valid = False
                   
    
    
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Video🔽", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='Detecting with 💕...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'data/videos/{uploaded_file.name}'
                
        else:
            is_valid = False

    if is_valid:
        print('valid')
        if st.button('Predict'):
            st.write("")
            if source_index == 0:
                with st.spinner(text='Preparing Images'):
                    for img in os.listdir(get_detection_folder()):
                        st.image(str(Path(f'{get_detection_folder()}') / img),caption='Detected Image')

                #PLOT
                img=Image.open(uploaded_file)
                batch_t = preprocess_ImageNet(img)

                #--------------------------------------------------------#

                # Load resnet architecture and download weights
                model = models.resnet101(pretrained=True)  # Trained on ImageNet
                model.eval()
            


                #--------------------------------------------------------#

                # Inference
                output = model(batch_t)
                detect(opt)
            
                pred = postprocess(output, topPred=5)
                st.write(pred)
                fig = px.bar(x=pred['class'], y=pred['conf'])
                st.plotly_chart(fig)

           
            else:
                with st.spinner(text='Preparing Video'):
                    for vid in os.listdir(get_detection_folder()):
                        st.video(str(Path(f'{get_detection_folder()}') / vid))
                        

