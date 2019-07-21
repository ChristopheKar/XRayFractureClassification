# -*- coding: utf-8 -*-

import os
import shutil
import pandas as pd


# =============================================================================
#                              Auxiliary functions
# =============================================================================


def copy_data(src,dst,file = False):
    
    '''
    file : 
    * True = copy file
    * false = copy all folder
    '''
    if not os.path.exists(dst):
        if file == False:
            shutil.copytree(src,dst,symlinks=True)
    
    if file == True:
        shutil.copy(src,dst)
            
def get_csv(dataset_path,dest):
    
    for i in os.listdir(dataset_path):
        if "csv" in i :
            copy_data(os.path.join(dataset_path,i),dest,file = True)
            


def create_folder(path):
    if not os.path.exists(path):
        print("Not exist")
        os.mkdir(path)
    

def get_main_f(folder_list):
    
    """
    Get list of folders in directory to variable
    
    Input : list of directories
    
    Output : List of directories in variable
    
    """
    
    f_list = []
    
    for i in folder_list:
        f_list.append(os.path.join("/home/michal/Pulpit/MURA/MURA-v1.1",i))
        print(f_list)
        
    return f_list


def read_csv(csv_list):
    
    
    """
    Get variables contains csvs in Dataframe
    
    Input : list of csv
    
    Output : csv in variable
    
    """
    
    # Generator
    results = []
    i = 0
    while i < len(csv_list):
        results.append(pd.read_csv(csv_list[i]))
        i += 1
    return results



def get_selected_feature(img_path_array,feature_path):
    
    """
    Get hand data from all features 
    
    """
    
    data = []
    
    for i in img_path_array["img"]:
        if feature_path in i:
            data.append(i)
            
    return data

def get_data_distribution(data):
    

    distr = {}
    negative = []
    positive = []
    
    for i in data:
        if "negative" in i:
            negative.append(i)
        elif "study1_positive":
            positive.append(i)
    
    distr["positive"] = positive
    distr["negative"] = negative
    
    return distr


def re_path(patch_folder):
    
    """
    
    Return array of corrected pathes of images
    
    """
    
    repath = []
    
    for i in patch_folder:
        repath.append(str("/home/michal/Pulpit/MURA/"+i))
    return repath

def separate_images(img_path,abnormal_class,normal_class):
    
    '''
    Separate images to folder class
    
    '''
    
    n = 0 #negative index
    p = 0 #positive index
    
    for i in img_path:
        print(i)
        if os.path.exists(i):
            if "negative" in i:
                n += 1
                src = i
                dest = (normal_class)
                shutil.copy(src,dest+"/normal"+str(n)+".png")
                print("negative")
            
            elif "positive" in i:
                p += 1
                src = i
                dest = (abnormal_class)
                shutil.copy(src,dest+"/abnormal"+str(p)+".png")
                print("positive")
             
                
def move_pictures_to_folder(src,dst,n_pictures,print_status = False):
    
    n = n_pictures

    if print_status:
        print("Move function: \n" )
        print("\nBefore")
        print("Source images : ", len(os.listdir(src)))
        print("Destination images : ",len(os.listdir(dst)))
    
    for i in reversed(sorted(os.listdir(src))):
        shutil.move(src + "/" + str(i),dst)
        n = n - 1
        if n == 0:
            break
    
    if print_status:
        print("\nAfter")
        print("Source images : ", len(os.listdir(src)))
        print("Destination images : ",len(os.listdir(dst)))
        

       
    
    
        

