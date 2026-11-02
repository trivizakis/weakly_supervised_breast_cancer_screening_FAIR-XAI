#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:18:31 2019

@author: eleftherios
"""
import sys
import numpy as np
# import tensorflow.keras as keras

from tensorflow.keras.models import Model
#from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Input
#from matplotlib.pyplot import imshow
#import glob
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import scale#,normalize

from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, roc_curve, auc
#from pandas import read_table as rt
#import pickle as pkl

from sklearn.feature_selection import f_classif as fc
from sklearn.feature_selection import SelectKBest as kbest

#Xception: 22M, L126
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xc

#VGG: M138-143, L23-26
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg

#ResNet: M25-60
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet import ResNet101
from tensorflow.keras.applications.resnet import ResNet152
from tensorflow.keras.applications.resnet import preprocess_input as preprocess_input_resnet

from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.applications.resnet_v2 import ResNet101V2
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_resnet_v2

#INC3: M23
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_inc3

#INCR2: M55
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_input_v2

#Mobile: M4
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_input_mobile

#DenseNet: M8-20
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.densenet import preprocess_input as preprocess_input_densenet

#NASNet: M5-88
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.nasnet import NASNetLarge
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nas

##3D models
# sys.path.append('../keras_3d_applications')

#Inceptions
from keras_3d_applications.inception_v3_3d import InceptionV3_3d as inc3d

#Xception
from keras_3d_applications.xception_3d import Xception_3d as xc3d

#DenseNets
from keras_3d_applications.densenet_3d import DenseNet3DSmall as dense3d

#ResNets
from keras_3d_applications.resnet50_3d import ResNet50_3d as res503d

#ResNetsV2
from keras_3d_applications.inception_resnet_v2_3d import InceptionResNetV2_3d as incr3d

#MobileNets
from keras_3d_applications.mobilenet_3d import MobileNet_3d as mob3d
from keras_3d_applications.mobilenet_v2_3d import MobileNetV2_3d as mob23d
# from keras_3d_applications.mobilenet_v3_3d import MobileNetV3Small as mob33d

def model_preprocess(image,model_name):
    if model_name == "inc3":
        image = preprocess_input_inc3(image)
    elif model_name == "vgg":
        image = preprocess_input_vgg(image)
    elif model_name == "incr2":
        image = preprocess_input_v2(image)
    elif model_name == "densenet":
        image = preprocess_input_densenet(image)
    elif model_name == "small_densenet":
        image = preprocess_input_densenet(image)
    elif model_name == "medium_densenet":
        image = preprocess_input_densenet(image)
    elif model_name == "nasnet":
        image = preprocess_input_nas(image)
    elif model_name == "mobile":
        image = preprocess_input_mobile(image)
    elif model_name == "xception":
        image = preprocess_input_xc(image)
    elif model_name == "resnet50":
        image = preprocess_input_resnet(image)        
    elif model_name == "resnet101":
        image = preprocess_input_resnet(image)  
    elif model_name == "resnet152":  
        image = preprocess_input_resnet(image)  
    elif model_name == "resnetv250":
        image = preprocess_input_resnet_v2(image)
    elif model_name == "resnetv2101":
        image = preprocess_input_resnet_v2(image)
    elif model_name == "resnetv2152":
        image = preprocess_input_resnet_v2(image)
    return image
        

def model_selection(input_shape, model_name, pooling, freeze_up_to, include_top, classes, volumetric=False, weights=None):
    #load pre-trained model
    print(input_shape)
    if not volumetric:
        if type(input_shape) == tuple:
            input_tensor = Input(shape=(input_shape[0], input_shape[1], input_shape[2]))
        else:
            input_tensor = input_shape
        
        if model_name == "xception":
            pretrained_model = Xception(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "inc3":
            pretrained_model = InceptionV3(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "vgg16":
            pretrained_model = VGG16(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes) 
        elif model_name == "vgg19":
            pretrained_model = VGG19(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes) 
        elif model_name == "incr2":
            pretrained_model = InceptionResNetV2(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "densenet201":
            pretrained_model = DenseNet201(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "densenet169":
            pretrained_model = DenseNet169(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "densenet121":
            pretrained_model = DenseNet121(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "nasnetm":
            pretrained_model = NASNetMobile(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "nasnetl":
            pretrained_model = NASNetLarge(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "mobile":
            pretrained_model = MobileNet(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "mobilev2":
            pretrained_model = MobileNetV2(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "resnet50":
            pretrained_model = ResNet50(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "resnet101":
            pretrained_model = ResNet101(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "resnet152":
            pretrained_model = ResNet152(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "resnetv250":
            pretrained_model = ResNet50V2(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "resnetv2101":
            pretrained_model = ResNet101V2(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "resnetv2152":
            pretrained_model = ResNet152V2(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        else:
            print("Model "+model_name+" does not exist.")

    else:
        input_tensor = Input(shape=(input_shape[0], input_shape[1], input_shape[2], 1))

        if model_name == "inc3d":
            pretrained_model = inc3d(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "incr3d":
            pretrained_model = incr3d(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "dense3d":
            pretrained_model = dense3d(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "xc3d":
            pretrained_model = xc3d(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "res503d":
            pretrained_model = res503d(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "mob3d":
            pretrained_model = mob3d(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        elif model_name == "mob23d":
            pretrained_model = mob23d(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
        # elif model_name == "mob33d":
        #     pretrained_model = mob33d(input_tensor=input_tensor, include_top = include_top, weights = weights, pooling = pooling, classes=classes)
    
    model = Model(inputs = pretrained_model.input,
                         outputs = pretrained_model.output)

    # i.e. freeze all convolutional InceptionV3 layers
    for layer in pretrained_model.layers[:freeze_up_to]:
        layer.trainable = False
    return model

def multi_classification(deep_features, labels, kernel="rbf", n_splits=5):
    #classification
    performance = []
    roc_auc_avg = []
    # auc_avg = []model_selection
    index=0
    skfold = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)
    for train_index,test_index in skfold.split(deep_features,labels):
        #split data
        X_train, X_test = deep_features[train_index],deep_features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        if kernel == "lr":
            clf = LogisticRegression(multi_class="ovr")
        else:
            clf = svm.SVC(kernel=kernel, probability=False)#poly, linear, rbf
            
        clf.fit(X_train, y_train)  
        score = clf.score(X_test,y_test)
        predictions = clf.predict(X_test)
        
        y_test = label_binarize(y_test, classes=np.unique(labels))
        ypreds = label_binarize(predictions, classes=np.unique(labels))
        
        roc = roc_auc_score(y_test,ypreds,average='macro',multi_class='ovr')
                
        index+=1
        print("Acc of fold "+str(index)+": "+str(score*100)+"%")
        print("Roc of fold "+str(index)+": "+str(roc*100)+"%")
        
        performance.append(score)    
        roc_auc_avg.append(roc)
        
    #final avg metrics
    fold_mean_acc = np.array(performance).mean()
    fold_mean_roc = np.array(roc_auc_avg).mean()
    fold_std_acc = np.array(performance).std()
    fold_std_roc = np.array(roc_auc_avg).std()
    print("Average Acc: "+str(fold_mean_acc*100)+"%")
    print("Average Roc: "+str(fold_mean_roc*100)+"%")
    
    return {"mean_acc":fold_mean_acc,"std_acc":fold_std_acc,"mean_roc":fold_mean_roc,"std_roc":fold_std_roc}
    
def binary_classification(deep_features, labels, kernel="rbf", n_splits=5):
    #classification
    performance = []
    roc_auc_avg = []
    auc_avg = []
    index=0
    skfold = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)
    for train_index,test_index in skfold.split(deep_features,labels):
        #split data
        X_train, X_test = deep_features[train_index],deep_features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        if kernel == "lr":
            clf = LogisticRegression()
        else:
            clf = svm.SVC(kernel=kernel, probability=False)#poly, linear, rbf
            
        clf.fit(X_train, y_train)  
        score = clf.score(X_test,y_test)
        predictions = clf.predict(X_test)
        
        roc = roc_auc_score(y_test,predictions[:,1],average='weighted')        
        fpr, tpr, thresholds = roc_curve(y_test,predictions[:,1])#, pos_label=1)
        auc_score = auc(fpr, tpr)
                
        index+=1
        print("Acc of fold "+str(index)+": "+str(score*100)+"%")
        print("Roc of fold "+str(index)+": "+str(roc*100)+"%")
        
        performance.append(score)    
        roc_auc_avg.append(roc)
        auc_avg.append(auc_score)
        
    #final avg metrics
    fold_mean_acc = np.array(performance).mean()
    fold_mean_roc = np.array(roc_auc_avg).mean()
    fold_std_acc = np.array(performance).std()
    fold_std_roc = np.array(roc_auc_avg).std()
    print("Average Acc: "+str(fold_mean_acc*100)+"%")
    print("Average Roc: "+str(fold_mean_roc*100)+"%")

    return {"mean_acc":fold_mean_acc,"std_acc":fold_std_acc,"mean_roc":fold_mean_roc,"std_roc":fold_std_roc}
    
    #please save a performance_report.txt
    #roc curves
def patient_based_classification(hypes, deep_features, labels_patient, labels_slice, kernel="rbf", n_splits=5):
    #classification (patient exam --> single image)
    performance = []
    roc_auc_avg = []
    
    keys = np.array(list(labels_patient.keys()), dtype=str)
    labels_ = np.array(list(labels_patient.values()), dtype=int)
    
    index=0
    skfold = StratifiedKFold(n_splits=n_splits, random_state=None, shuffle=True)
    
    ksplits=[]
    for train_index,test_index in skfold.split(keys,labels_):
        X_train, X_test = keys[train_index],keys[test_index]
        ksplits.append([X_train, X_test])
    
    for index,split_ in enumerate(ksplits):
        X_train = []
        y_train = []
        X_test = []
        y_test = []
        
        #training set
        for patient in split_[0]:
            for slice_ in list(labels_slice.keys()):
                if patient in slice_:
                    X_train.append(deep_features[slice_])
                    y_train.append(labels_slice[slice_])
                    
        #testing set
        for patient in split_[1]:
            for slice_ in list(labels_slice.keys()):
                if patient in slice_:
                    X_test.append(deep_features[slice_])
                    y_test.append(labels_slice[slice_])
        
        print("# samples:")
        print(len(X_train))
        print(len(y_train))
        print(len(X_test))
        print(len(y_test))
        X_train = np.array(X_train)
        y_train = np.stack(y_train)
        X_test = np.array(X_test)
        y_test = np.stack(y_test)
        
        if kernel == "lr":
            clf = LogisticRegression()# default auto == "ovr"
        else:
            clf = svm.SVC(kernel=kernel)#, probability=True)#poly, linear, rbf
        clf.fit(X_train, y_train)  
        score = clf.score(X_test,y_test)
        # predictions = clf.predict_proba(X_test)
        predictions = clf.predict(X_test)
        
        if hypes["num_classes"]>2:
            y_test = label_binarize(y_test, classes=np.unique(labels_))
            ypreds = label_binarize(predictions, classes=np.unique(labels_))
        else:
            ypreds = predictions
        
        roc = roc_auc_score(y_test,ypreds,average='macro',multi_class='ovr')        
        # fpr, tpr, thresholds = roc_curve(y_test,predictions[:,1])#, pos_label=1)
                
        index+=1
        print("Acc of fold "+str(index)+": "+str(score*100)+"%")
        print("Roc of fold "+str(index)+": "+str(roc*100)+"%")
        
        performance.append(score)    
        roc_auc_avg.append(roc)
        
    #final avg metrics
    fold_mean_acc = np.array(performance).mean()
    fold_mean_roc = np.array(roc_auc_avg).mean()
    fold_std_acc = np.array(performance).std()
    fold_std_roc = np.array(roc_auc_avg).std()
    print("Average Acc: "+str(fold_mean_acc*100)+"%")
    print("Average Roc: "+str(fold_mean_roc*100)+"%")
    
    return {"mean_acc":fold_mean_acc,"std_acc":fold_std_acc,"mean_roc":fold_mean_roc,"std_roc":fold_std_roc}

def classify(args, deep_features, labels):
    #classification

    #split data
    X_train, X_test = deep_features[args["training_set"]],deep_features[args["testing_set"]]
    y_train, y_test = labels[args["training_seweightst"]], labels[args["testing_set"]]
    
    clf = svm.SVC(kernel=args["kernel"], gamma='scale', probability=True)#poly, linear, rbf
    clf.fit(X_train, y_train)
    score = clf.score(X_test,y_test)
    predictions = clf.predict_proba(X_test)
    
    roc = roc_auc_score(y_test,predictions[:,1],average='weighted')        
    fpr, tpr, thresholds = roc_curve(y_test,predictions[:,1])#, pos_label=1)
    auc_score = auc(fpr, tpr)
        
    print("ACC: "+str(score*100)+"%")
    print("ROC: "+str(roc*100)+"%")
    print("AUC: "+str(auc_score*100)+"%")
    return score, roc, auc_score
    
def preprocessing(deep_features, labels, model_name):
    #labels: list
    #deep_features: list
    deep_features_raw, labels=shuffle(deep_features,labels)

    #feature preprocessing
    if model_name == "inc3":
        deep_features=np.array(deep_features_raw).reshape(-1,2048)#INc3
    elif model_name == "vgg":
        deep_features=np.array(deep_features_raw).reshape(-1,512)#VGG19
    elif model_name == "incr2":
        deep_features=np.array(deep_features_raw).reshape(-1,1536)#InceptionResNetV2 #[:512]
    elif model_name == "densenet":
        deep_features=np.array(deep_features_raw).reshape(-1,1920)#DenseNet201
    elif model_name == "nasnet":
        deep_features=np.array(deep_features_raw).reshape(-1,4032)#NASNetLarge
        
    #feature normalization    
    deep_features = scale(deep_features,with_mean=True,with_std=True)#the best
    #deep_features = normalize(deep_features)#big no-no
    
    #labels: list to npy 
    labels=np.array(labels)
    
    return deep_features, labels

class Transferable_Networks:    
    def __init__(self,hypes):
        self.hypes = hypes
    
    def get_pretrained(self,input_shape, model_name, pooling, freeze_up_to, include_top, classes, volumetric, weights):
        return model_selection(input_shape=input_shape,model_name=model_name, pooling=pooling, freeze_up_to=freeze_up_to, include_top=include_top, classes=classes, volumetric=volumetric, weights=weights)

#old:
#     def extract_features(self, images, labels, model_name="vgg"):
                
# #        model_initialization(model_name=model_name)
#         model = model_selection(self.hypes["input_shape"], model_name=model_name)
#         #infer
#         deep_features=[]
#         for image in images:
#             if self.hypes["input_shape"][2]==1:
#                 image = np.concatenate((image,image,image),axis=2)
#             image = np.expand_dims(image, axis=0)
#             image = model_preprocess(image,model_name=model_name)
#             deep_features.append(model.predict(image))
            
#         return preprocessing(deep_features, labels, model_name=model_name)
#new:
    def extract_features(self, images, model_name="vgg",threshold=None):
        #images: dict
        model = model_selection(self.hypes["input_shape"], model_name=model_name,pooling=self.hypes["pooling"], freeze_up_to=-1, include_top=False, classes=self.hypes["num_classes"], weights="imagenet")
        #infer
        deep_features={}
        for img_name in list(images.keys()):
            image = images[img_name]
            if self.hypes["input_shape"][2]==1:
                image = np.concatenate((image,image,image),axis=2)
            image = np.expand_dims(image, axis=0)
            deep_features[img_name]=model.predict(image).ravel()
            
        deep_features_list = list(deep_features.values())
        deep_features_keys = list(deep_features.keys())
        
        deep_features_array = preprocessing(deep_features_list, model_name=model_name, normalization=self.hypes["normalization"], threshold=threshold)
        
        final_features={}
        for index,key in enumerate(deep_features_keys):
            final_features[key] = deep_features_array[index]
                        
        return final_features
    
    def extract_features_from_image_list(self, images, model_name="vgg",threshold=None):
        #images list
                
        model = model_selection(self.hypes["input_shape"], model_name=model_name,pooling=self.hypes["pooling"], freeze_up_to=-1, include_top=False, classes=self.hypes["num_classes"], weights="imagenet")
        #infer
        deep_features=[]
        for image in images:
            image = np.expand_dims(image, axis=0)
            deep_features.append(model.predict(image).ravel())
        
        deep_features = preprocessing(deep_features, model_name=model_name, normalization=self.hypes["normalization"], threshold=threshold)
                    
        return deep_features

    def select_features(self, k, deep_features, labels):
#        X = np.array(deep_features)
#        y = np.reshape(np.array(labels),(-1,))
        X=deep_features
        y=labels
        #feature selection
        selector = kbest(fc, k=k)
        best_features = selector.fit_transform(X, y)
        
        f_scores, p_values = fc(X, y)
        print("Number of statistical significant features: "+str(len(p_values[p_values<0.005])))
        
        return best_features
         
    def simple_classifier(self, args, deep_features, labels):
        #args={}
        #args["kernel"]
        #args["training_set"]
        #args["testing_set"]
        return classify(args, deep_features, labels)
   
    def classification(self,args, deep_features, labels):
        if args["num_classes"]==2:
            metrics = binary_classification(deep_features, labels, kernel=args["kernel"], n_splits=args["folds"])
        else:
            metrics = multi_classification(deep_features, labels, kernel=args["kernel"], n_splits=args["folds"])
        return metrics
            
    def classification_per_patient(self,args, deep_features, labels_patient, labels_slice):
        return patient_based_classification(self.hypes, deep_features, labels_patient, labels_slice, kernel=args["kernel"], n_splits=args["folds"]) 