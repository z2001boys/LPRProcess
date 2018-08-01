import os
import cv2 as cv 
import numpy as np

alphabet = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'A': 10, 'B': 11,
            'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'J': 18, 'K': 19, 'L': 20, 'M': 21, 'N': 22,
            'P': 23, 'Q': 24, 'R': 25, 'S': 26, 'T': 27, 'U': 28, 'V': 29, 'W': 30, 'X': 31, 'Y': 32, 'Z': 33}

# Character to number.
def char_to_num(char):
    num = alphabet[char]
    return num

# Number to character.
def num_to_char(num):
    for key in alphabet:
        if alphabet[key] == num:
            return key

# Generate the data label file 
# Input: image dataset path 
# Output: image label file path (ex: /home/output.txt)
def extract_label_file(dir_path, label_path):
    if (os.path.isfile(label_path)):
        os.remove(label_path)
  
    for img_path in os.listdir(dir_path):
        filename = dir_path + '/' + img_path
        label = img_path[:1]
        
        line = filename + ',' + label + '\n'
        
        with open(label_path, 'a') as f:
            f.write(line)

# Load the label from label.txt
# Input: label path (ex: output.txt)
# Output: image array (numpy array) and label number (numpy array)
def load_from_label_text(label_path):
    img_list = []
    label_list = []
    
    with open(label_path, 'r') as f:
        for line in f.readlines():
            img_path = line.split(',')[0]
            label = line.split(',')[1].split('\n')[0]
            
            img = np.asarray(cv.imread(img_path, 0))
            label_num = char_to_num(label)
            
            img_list.append(img)
            label_list.append(label_num)
            
    np_data_x = np.array(img_list)
    np_data_y = np.array(label_list)
    
    return np_data_x, np_data_y
