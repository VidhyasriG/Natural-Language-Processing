#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Course: AIT 690
Assignment: Programming Assignment 2 - POS Tagger
Date: 03/19/2019
Team Name: ' A team has no name'
Members:    1. Rahul Pandey
            2. Arjun Mudumbi Srinivasan
            3. Vidhyasri Ganapathi

***********************************************************************************************************************
tagger.py is a python program that assigns parts of speech tags maximising p(tag|word) to words in a
training file. The tag 'NN' is assigned for the words that are not in the training file. Words which only have
one part of speech in the training data are labeled with that same tag in the test file. Words with multiple parts
of speech which have unlabeled neighbors are tagged as their most likely tag in the training data set.
The untagged words with tagged neighbors were assigned based on maximizing their conditional probabilities.
To estimate the correct tag, we first calculate the probability of the word, given tag is calculated as
P(word|tag)=[freq(tag,word)/freq(tag)].
Next, we calculate the probability of the tag, given previous tag - P(tag | prevTag)=[freq(prevTag, tag)/freq(prevTag)]
The accuracy of the model before adding rules is ** 85.68% **. After adding the rules, the accuracy of the model has
increased to ** 90.66% **

*******************************************************************
Following are the rules added:
Rule 1: Tag every word that contains number to "CD"
Rule 2: If current word is tagged DT and previous word is "all", change the tag of "all" to "PDT" (Pre Determiner)
Rule 3: If current word is tagged NN i.e. singular noun and the word is capitalized, change the tag to "NNP" i.e. Proper Noun
Rule 4: If current word is tagged VBN i.e. past participle verb and previous word was capitalized then change the current tag to VBD i.e. past tense verb
Rule 5: If current word is tagged VB and previous word was tagged determiner, then change the current tag to NN i.e. singular noun
Rule 6: If current word is tagged NN and previous word was tagged TO i.e. to, then change the current tag to VB i.e. verb
Rule 7: If current word is tagged NN and previous word was tagged MD i.e. Modal, then change the current tag to VB i.e. verb
*******************************************************************

*******************************************************************
Following are the data used -
The labeled training data is "pos-train.txt"
The untagged test file is "pos-test.txt"
The predicted labeled test data is "pos-test-with-tags.txt"
The golden standard labeled test data is "pos-test-key.txt"
The scoring file is "scorer.py"
"pos-tagging-report.txt" and "tagger-log.txt" are logging and reporting files
*******************************************************************

To run the script enter the following command -
$ python tagger.py pos-train.txt pos-test.txt > pos-test-with-tags.txt
***************************************************************************************
"""

# import statements
import numpy as np
import pandas as pd
import operator
import re
import nltk
import sys
from nltk.probability import FreqDist
from nltk.probability import ConditionalFreqDist

# Method to get training data
# Input: file_path path of the training file taken from arguments
# Return: A list of tuples where each tuples represent word and its associated tags
def get_training_data(file_path):
    tokens = []
    for lines in open(file_path):
        lines = lines.strip()
        if lines.startswith("["):
            lines = lines[2:-2] # remove the starting and trailing braces [ ]
        if "\\/" in lines:
            lines = lines.replace("\\/", "_slash_") # Replace "\\/" with _slash_ token so that it doesn't be the part of word-tag split
        lines = lines.split(" ")
        for tkn in lines:
            if "|" in tkn:
                tkn = tkn.split("|")[0] # if | is present only take the first element
            if tkn!="":
                tkns = nltk.tag.str2tuple(tkn)
                if len(tkns) == 2:
                    tokens.append((tkns[0].replace("_slash_", "/"), tkns[1])) # restore the / by replacing the _slash_
    tokens = np.array(tokens)
    return tokens

# Method to get testing data
# Input: file_path path of the testing file taken from arguments
# Return: A list containing all the target test words
def get_test_data(file_path):
	test_tokens = []
	for lines in open("pos-test.txt"):
	    lines = lines.strip()
	    if lines.startswith("["):
	        lines = lines[2:-2] # remove the trailing and starting braces [ ]
	    if "\\/" in lines: # replace the \\/ with single slash as equivalent in training
	        lines = lines.replace("\\/", "/")
	    lines = lines.split(" ") # split by space
	    for tkns in lines:
	        if tkns != "": # don't take empty tokens
	            test_tokens.append(tkns)
	test_tokens = np.array(test_tokens)
	return test_tokens

# Method to get testing data
# Input:
# # word: current word to tag
# # prev_tag: the tag of the previous word
# # word_tag_freq_dist: the dict of dict containing the frequency of word and their tags occuring together
# # tag_tag_confidence: the dict of dict containing the frequency of previous tag to next tag occuring together
# # tag_freq_dist: the dict containing the frequency of each tags
# Return: The tag with highest score calculated based on P(tag|word)
def get_tagged(word, prev_tag, word_tag_freq_dist, tag_tag_confidence, tag_freq_dist):
	if word not in word_tag_freq_dist.keys():
		return "NN"
	else:
		if not prev_tag:
			return word_tag_freq_dist[word].max()
		else:
			scores = []
		for tag, w_t_score in word_tag_freq_dist[word].items():
			t_t_score = tag_tag_confidence[prev_tag][tag]
			p_t_t = t_t_score/float(max(tag_freq_dist[prev_tag], 0.5))
			p_w_t = w_t_score/float(max(tag_freq_dist[tag], 0.5))
			scores.append((tag, p_w_t*p_t_t))
		scores = sorted(scores, key=lambda x: x[1], reverse=True)
		return scores[0][0]


# Method to get testing data
# Input: file_path path of the testing file taken from arguments
# Return: A list containing all the target test words
def main():
	train_file_path = sys.argv[1] # Get the training file path from arguments
	tokens = get_training_data(train_file_path) # get all the word tag pair
	tag_freq_dist = FreqDist(tag for (word, tag) in tokens) # get the frequency of all tags
	word_tag_freq_dist = ConditionalFreqDist((word,tag) for word, tag in tokens) #
	proximity_pairs = nltk.bigrams(tokens) # compute the bigrams of tags
	tag_tag_confidence = nltk.ConditionalFreqDist((a[1], b[1]) for (a,b) in proximity_pairs) # compute frequency of occurance of prev tag to current tag
	test_file_path = sys.argv[2] # Get the test file path from arguments
	test_tokens = get_test_data(test_file_path) # get all the words of test
	result_tagged = [] # store each word and their associated tags
	for i in range(len(test_tokens)):
	    if i == 0: # when there is no previous tag
	        result_tagged.append((test_tokens[i], get_tagged(test_tokens[i], prev_tag=None
															, word_tag_freq_dist=word_tag_freq_dist
															, tag_tag_confidence=tag_tag_confidence
															, tag_freq_dist=tag_freq_dist)))
	    else:
	        result_tagged.append((test_tokens[i], get_tagged(test_tokens[i], result_tagged[i-1][1]
															, word_tag_freq_dist=word_tag_freq_dist
															, tag_tag_confidence=tag_tag_confidence
															, tag_freq_dist=tag_freq_dist)))
	        # Rule 1: Tag every word that contains number to "CD"
	        if re.match("(\d+(\.\d+)?)", test_tokens[i]) is not None:
	            result_tagged[-1] = (result_tagged[-1][0], "CD")
	        # Rule 2: If current tag is DT and previous word is "all", change the tag of "all" to "PDT" (Pre Determiner)
	        if (result_tagged[-1][1] == "DT" and test_tokens[i-1] == "all"):
	            result_tagged[-2] = ("all", "PDT")
	        # Rule 3: If current word is tagged NN i.e. singular noun and the word is capitalized, change the tag to "NNP" i.e. Proper Noun
	        if (result_tagged[-1][1] == "NN" and test_tokens[i][0].isupper()):
	            result_tagged[-1] = (result_tagged[-1][0], "NNP")
	        # Rule 4: If current word is VBN i.e. past participle verb and previous word was capitalized then change the current tag to VBD i.e. past tense verb
	        if (result_tagged[-1][1] == "VBN" and test_tokens[i-1][0].isupper()):
	            result_tagged[-1] = (result_tagged[-1][0], "VBD")
	        if len(result_tagged) >= 2:
	            # Rule 5: If current tag is VB and previous word was tagged determiner, then change the current tag to NN i.e. singular noun
	            if (result_tagged[-1][1] == "VB" and (result_tagged[-2][1] == "DT")):
	                result_tagged[-1] = (result_tagged[-1][0], "NN")
	            # Rule 6: If current tag is NN and previous word was tagged TO i.e. to, then change the current tag to VB i.e. verb
	            if (result_tagged[-1][1] == "NN" and (result_tagged[-2][1] == "TO")):
	                result_tagged[-1] = (result_tagged[-1][0], "VB")
	            # Rule 7: If current tag is NN and previous word was tagged MD i.e. Modal, then change the current tag to VB i.e. verb
	            if (result_tagged[-1][1] == "NN" and (result_tagged[-2][1] == "MD")):
	                result_tagged[-1] = (result_tagged[-1][0], "VB")

	for word, tag in result_tagged: # print all the result to STDOUT
		print("%s %s" % (word, tag))

if __name__=='__main__': #call the main method
	main()
