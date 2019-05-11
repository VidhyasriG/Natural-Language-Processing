"""
Course: AIT 690
Assignment: Programming Assignment 3 - WSD
Date: 04/04/2019
Team Name: ' A team has no name'
Members:    1. Rahul Pandey
          2. Arjun Mudumbi Srinivasan
          3. Vidhyasri Ganapathi

*****************************************************************************************
This code is used to evaluate the performance of the WSD decision-lits in the metrics of accuracy.
It will generate a file to report the accuracy and a confusion matrix.
One could run the scorer.py like:
$ python scorer.py my-line-answers.txt line-answers.txt
"""
"""
********************************************************************************************
Importing all the required library

"""
import pandas as pd
import sys
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# %%%%%%%%%%%%%
def main():
    predicted_file = sys.argv[1]
    gold_file = sys.argv[2]
    # Read the Gold standard file
    with open(gold_file, "r") as handler:
        handlerlist = []  # dumping all the yTrue into a list
        for line in handler:
            handlerlist.append(line)
    handlerlist = pd.Series(handlerlist)  # Converting into pandas series
    handlerlist = handlerlist.str.split('"', expand=True)  # split each entry at "
    handlerlist = handlerlist.iloc[:, -2]  # taking only the sense as the handlerlist
    with open(predicted_file, 'r') as handlerpred:  # Reading in the Predictions files
        handlerpredlist = []
        for line in handlerpred:
            handlerpredlist.append(line)
    handlerpredlist = pd.Series(handlerpredlist)  # Converting into pandas series
    handlerpredlist = handlerpredlist.str.split("'", expand=True)  # split each entry at "
    handlerpredlist = handlerpredlist.iloc[:, -2]  # taking only the sense as the handlerpredlist
    uni_gd_tags = list(set(handlerlist))  # getting the unique senses from the gold standard
    print(uni_gd_tags)
    cm = confusion_matrix(handlerlist, handlerpredlist)
    print("\t%s" % ("\t".join(uni_gd_tags)))
    for i in range(len(cm)):
        print("%s\t%s" % (
        uni_gd_tags[i], "\t".join(["%.3f" % (x) for x in cm[i]])))  # Confusion matrix print along with the tags
    accuracyscore = accuracy_score(handlerlist, handlerpredlist)
    print('accuracy score=%.3f%%' % (accuracyscore*100))  # print the accuracy score


if __name__ == '__main__':
    main()
