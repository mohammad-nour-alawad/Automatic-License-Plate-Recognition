import numpy as np
import os
import urllib.request
from matplotlib import gridspec
from matplotlib import pyplot as plt
from PIL import Image
import argparse
import sys
import math
from ksnn.api import KSNN
from ksnn.types import *
import cv2 as cv
import time
import imutils



from dotenv import load_dotenv, find_dotenv
import pprint
from datetime import datetime
from pymongo import MongoClient
import gridfs

import base64
from io import BytesIO
from PIL import Image

#load_dotenv(find_dotenv())


connection_string = "mongodb+srv://anpr:anpranpr@lpr.et0guo9.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(connection_string)
anpr_results = client.anpr_results

detection_collection = anpr_results.detection_collection
images_collection = anpr_results.images
cars_collection = anpr_results.cars_collection

#dbs = client.list_database_names()
#print(dbs)


GRID0 = 13
GRID1 = 26
#GRID2 = 52

LISTSIZE1 = 7
LISTSIZE2 = 15
SPAN = 3
NUM_CLS1 = 2
NUM_CLS2 = 10

MAX_BOXES = 500
OBJ_THRESH = 0.3
NMS_THRESH = 0.2

CLASSES1 = ("car","license")
CLASSES2 = ("0","1","2","3","4","5","6","7","8","9")

#CLASS = "Hand"

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])
    box_wh = np.exp(input[..., 2:4])
    box_wh = box_wh * anchors

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    #print(f"grid: {grid_h=}, {grid_w=}")
    box_xy += grid
    box_xy /= (grid_w, grid_h)
    box_wh /= (416, 416)
    box_xy -= (box_wh / 2.)
    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= OBJ_THRESH)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def handtiny_post_process(input_data):
    masks = [[3, 4, 5], [0, 1, 2]]
    anchors = [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169],
            [344, 319]]

    boxes, classes, scores = [], [], []
    for input,mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        
        print('class: {}, score: {}'.format(CLASSES1[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x+w, y+h))
        x = x * image.shape[1]
        y = y * image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv.putText(image, '{0} {1:.2f}'.format(CLASSES1[cl], score),
                    (top, left - 6),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def draw2(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box
        
        print('class: {}, score: {}'.format(CLASSES2[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(x, y, x+w, y+h))
        x = x * image.shape[1]
        y = y * image.shape[0]
        w *= image.shape[1]
        h *= image.shape[0]
        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv.putText(image, '{0} {1:.2f}'.format(CLASSES2[cl], score),
                    (top, left - 6),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--library", help="Path to C static library file")
    parser.add_argument("--model", help="Path to nbg file")
    parser.add_argument("--picture", help="Path to input picture")
    parser.add_argument("--level", help="Information printer level: 0/1/2")

    args = parser.parse_args()
    if args.model :
        if os.path.exists(args.model) == False:
            sys.exit('Model \'{}\' not exist'.format(args.model))
        model = args.model
        model_plates = model[:-7] + 'digits.nb'
    else :
        sys.exit("NBG file not found !!! Please use format: --model")
    if args.picture :
        if os.path.exists(args.picture) == False:
            sys.exit('Input picture \'{}\' not exist'.format(args.picture))
        picture = args.picture
    else :
        sys.exit("Input picture not found !!! Please use format: --picture")
    if args.library :
        if os.path.exists(args.library) == False:
            sys.exit('C static library \'{}\' not exist'.format(args.library))
        library = args.library
        library_plates = library[:-7] + 'digits.so'
    else :
        sys.exit("C static library not found !!! Please use format: --library")
    if args.level == '1' or args.level == '2' :
        level = int(args.level)
    else :
        level = 0

    handtiny = KSNN('VIM3')
    
    handtiny.nn_init(library=library, model=model, level=level)
    handtiny2 = KSNN('VIM3')
    
    handtiny2.nn_init(library=library_plates, model=model_plates, level=level)
    info2 = handtiny2.nn_get_output_tensor_info(num=0)


    cv_img =  list()
    img = cv.imread(picture, cv.IMREAD_COLOR)
    img_bac = np.array(img)
    cv_img.append(img)
    
    print('Start inference ...')
    start = time.time()

      
    info = handtiny.nn_get_output_tensor_info(num=0)
    
    data = handtiny.nn_inference(cv_img, platform='DARKNET', reorder='2 1 0', output_tensor=2 ,output_format=output_format.OUT_FORMAT_FLOAT32)
    end = time.time()
    
    
    print('inference : ', end - start)
    input0_data = data[0]
    input1_data = data[1]
    
    input0_data = input0_data.reshape(SPAN, LISTSIZE1, GRID0, GRID0)
    input1_data = input1_data.reshape(SPAN, LISTSIZE1, GRID1, GRID1)
    
    input_data = list()
    input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
    input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
    
    
    boxes, classes, scores = handtiny_post_process(input_data)
    

    if boxes is not None:
        print(boxes, img.shape)
        imggg = img_bac
        
        now = datetime.now()
        tmp = str(now)
        nowstr = ""
        for s in tmp:
        	if s!=' ':
        		nowstr+=s
        mx = 0
        for box, score, cl in zip(boxes, scores, classes):
            x, y, w, h = box
                        
            x = x * imggg.shape[1]
            y = y * imggg.shape[0]
            w *= imggg.shape[1]
            h *= imggg.shape[0]
            top = max(0, np.floor(x + 0.5).astype(int))
            left = max(0, np.floor(y + 0.5).astype(int))
            right = min(imggg.shape[1], np.floor(x + w + 0.5).astype(int))
            bottom = min(imggg.shape[0], np.floor(y + h + 0.5).astype(int))
            
            if cl == 1:
                print(y, y+h, x, x+w)
                part = imggg[int(y):int(y+h) , int(x):int(x+w)]
                
                print(score)
                print(cl)
                mx = max(mx,score)
                plates = list()
                
                if part.shape[0]==0:
                	continue;
                
                
                print(top,bottom,left,right,score)
                
                
                print(part.shape)
                part = rotate_image(part, -10)
                
                
                imgg_for_database = np.array(part)
                
                part = cv.resize(part, (800, 300), interpolation = cv.INTER_AREA)
                
                plates.append(part)
                
                
                data2 = handtiny2.nn_inference(plates, platform='DARKNET', reorder='2 1 0', output_tensor=2 ,output_format=output_format.OUT_FORMAT_FLOAT32)
                
                input0_data2 = data2[0]
                input1_data2 = data2[1]
                
                
                input0_data2 = input0_data2.reshape(SPAN, LISTSIZE2, GRID0, GRID0)
                input1_data2 = input1_data2.reshape(SPAN, LISTSIZE2, GRID1, GRID1)
                
                input_data2 = list()
                
                input_data2.append(np.transpose(input0_data2, (2, 3, 0, 1)))
                input_data2.append(np.transpose(input1_data2, (2, 3, 0, 1)))
                
                boxes2, classes2, scores2 = handtiny_post_process(input_data2)
                
                if boxes2 is not None:
                
                    print(boxes2, part.shape)
                    boxes_and_cls = []
                    final_accuracy = 0.
                    for ind, bx in enumerate(boxes2):
                    	bxx = []
                    	final_accuracy += scores2[ind]
                    	
                    	for bb in bx:
                    		bxx.append(bb)
                    	bxx.append(classes2[ind])
                    	boxes_and_cls.append(bxx)
                    
                    boxes_and_cls = sorted(boxes_and_cls, key = lambda x : x[0])
                    detection = ""
                    final_accuracy /= max(6,len(scores2))
                    
                    for cc in boxes_and_cls:
                    	detection += str(cc[4])
                    
                    print(detection)
                    
                    encoded1,buffer1 = cv.imencode('.jpg',imgg_for_database,[cv.IMWRITE_JPEG_QUALITY,30])
                    b64 = base64.b64encode(buffer1).decode("utf-8")
                    
                    doc = {"time" : datetime.now().strftime("%d/%m/%Y, %H:%M:%S"), "license_number" : detection, "accuracy" : final_accuracy, 'filename' : nowstr, 'cam' : 'cam1'}
                    
                    insert_id = detection_collection.insert_one(doc)
                    images_collection.insert_one({'filename' : nowstr, 'data' : b64});
                    cv.imwrite("plates_detections/" + nowstr + ".png", part)
                    
                    
                    print("ok")
                    draw2(part, boxes2, scores2, classes2)
                    break
        
        
            if cl == 0:
            	part2 = imggg[max(int(y),0):min(int(y+h),960) , max(int(x),0):min(int(x+w), 1280)]
            	
            	encoded,buffer = cv.imencode('.jpg',part2,[cv.IMWRITE_JPEG_QUALITY,30])
            	b64 = base64.b64encode(buffer).decode("utf-8")
            	
            	
            	cars_collection.insert_one({'filename' : nowstr, 'data' : b64});
            	cv.imwrite("cars_detections/" + nowstr + ".png", part2)
        
        draw(img, boxes, scores, classes)



    cv.imshow("capture", img)
    cv.imshow("plate", part)
    cv.waitKey(0) 
    
    cv.destroyAllWindows() 
