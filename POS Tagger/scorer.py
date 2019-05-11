"""
Course: AIT 690
Assignment: Programming Assignment 2 - POS Tagger
Date: 03/19/2019
Team Name: ' A team has no name'
Members:    1. Rahul Pandey
            2. Arjun Mudumbi Srinivasan
            3. Vidhyasri Ganapathi

***********************************************************************************************************************
scorer.py is a python program that takes in two input . One should have the generated tags of the text and the other should have the
ground truth. The file returns the Confusion Matrix and accuracy for the predicted tags and ground truth tags. This will be stored in the pos-tagging-report.txt
***********************************************************************************************************************
To run the script enter the following command -
$ python scorer.py pos-test-with-tags.txt pos-test-key.txt > pos-tagging-report.txt
"""
#  Importing all the necessary libraries needed for the scorer
import numpy as np
import pandas as pd
import operator
import re
import nltk
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
from nltk.probability import ConditionalFreqDist
from sklearn.metrics import confusion_matrix

"""
function used for the generating the ground truth data and for cleaning the data .
The function strips the data and the removes the "[]".The function returns a list of list containing the word and the tags
=
"""
def get_ground_truth_data(file_path):
    tokens = []
    for lines in open(file_path):
        lines = lines.strip()
        if lines.startswith("["):
            lines = lines[2:-2]
        if "\\/" in lines:
            lines = lines.replace("\\/", "_slash_")
        lines = lines.split(" ")
        for tkn in lines:
            if "|" in tkn:
                    tkn = "|".join(tkn.split("|")[:-1])
            if tkn!="":
                tkns = nltk.tag.str2tuple(tkn)
                if len(tkns) == 2:
                    tokens.append((tkns[0].replace("_slash_", "/"), tkns[1]))
    tokens = np.array(tokens)
    return tokens

# """"
#  Function is used for generating the ground truth from the pred_token.txt file
#  The function returns a list containing the word and predicted tags
#
# """"
def get_predicted_data(file_path):
    tokens = []
    for lines in open(file_path):
        lines = lines.strip()
        tokens.append(lines.split(" "))
    return tokens
# """
# We are using the scikit learns implementation to generate the confusion matrix
# """
def get_cm(gt_tokens,pred_tokens,unique_gt_tags):
	cm=confusion_matrix(gt_tokens, pred_tokens, labels=unique_gt_tags)
	return cm

pred_file_path = sys.argv[1] # reqading the pred file
pred_tokens = get_predicted_data(pred_file_path)

gt_file_path = sys.argv[2] #  reading the ground truth
gt_tokens = get_ground_truth_data(gt_file_path)

cnt_total = 0
for i in range(len(gt_tokens)):# #  We are checking where the predicted tokens and ground truth are the same
    if gt_tokens[i][0] == pred_tokens[i][0]:
        cnt_total += 1

# Calculating the accuracy of our predicted results
if cnt_total == len(gt_tokens):
    cnt = 0
    for i in range(len(gt_tokens)):
        if gt_tokens[i][1] == pred_tokens[i][1]:
            cnt += 1
    print("Accuracy %.2f %%" % ((100.0*cnt)/len(gt_tokens)))# print the accuracy for tagger

# Now calculating confusion matrix
pred_token_tags=[]
gt_token_tags=[]
for word,tag in pred_tokens:
	pred_token_tags.append(tag)
for word,tag in gt_tokens:
	gt_token_tags.append(tag)
unique_gt_tags=list(set(gt_token_tags))
from sklearn.metrics import accuracy_score
print("Accuracy sklearn = %.3f %%" % (accuracy_score(gt_token_tags, pred_token_tags)))
cm=get_cm(gt_token_tags,pred_token_tags,unique_gt_tags=unique_gt_tags)# calling the confusion matric and printing in the console
print("\t%s"%("\t".join(unique_gt_tags)))
for i in range(len(cm)):
	print("%s\t%s" %(unique_gt_tags[i],"\t".join(["%.3f" %(x) for x in cm[i]])))
