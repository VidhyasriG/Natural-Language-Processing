"""
Course: AIT 690
Assignment: Programming Assignment 3 - WSD
Date: 04/04/2019
Team Name: ' A team has no name'
Members:    1. Rahul Pandey
          2. Arjun Mudumbi Srinivasan
          3. Vidhyasri Ganapathi
decision-list.py program implements a decision list classifier to perform word sense disambiguation
on the word 'line' used in different contexts.
Feature implemented from Yarowsky paper:
    1) index = -1word from target
    2) index = +1word from target
    3) index = -1 and -2 words from target
    4) index = +1 and +2 words from target
    5) index = -K words from target (k=3)
    6) index = +K words from target (k=3)

The program learns a decision list from line-train.xml. Now the decision list is applied to each
of the sentences learned from xml to assign a sense. The output is stored to my-decision-list.txt.
The list shows each feature, the log-likelihood score associated with it, and the sense it predicts.
The program outputs the answer tags it creates for each sentence to
STDOUT.

Files used:

line-train.xml contains examples of the word line used in the sense of a phone line and a product
line where the correct sense is marked in the text (to serve as an example from which to learn).
linetest.xml contains sentences that use the word line without any sense being indicated, where the
correct answer is found in the file line-answers.txt.
decision-list.py learns its decision list from line-train.xml and then
apply it to line-test.xml.
scorer.py  will take as input your sense tagged output and compare it with the gold
standard "key" data in line-answers.txt.

Our performance = 88.889%


Baseline performance assuming all tags are the 'phone' sense = 57.15%   = 72/126
Our Confusion Matrix:
           phone  product
phone        66      6
product      8       46


Run the prediction file:
$ python decision-list.py line-train.xml line-test.xml > my-line-answer.txt
Run the scoring file:
$ python scorer.pl my-line-answers.txt line-answers.txt

"""
from collections import defaultdict
from xml.dom.minidom import parse
import xml.dom.minidom
from nltk.tokenize import word_tokenize
import math
import sys
import operator


# Method to get training data
# Input: file_path path of the training file taken from arguments
# Return: A Python Dictionary train_data, which contains for each instance id,
#         its target sense label and all the context sentence where the target word appears
def get_training_data(file_path):
    DOMTree = xml.dom.minidom.parse(file_path)
    collection = DOMTree.documentElement
    context_list = collection.getElementsByTagName("context")  # list of all context sentences
    id_sense_list = collection.getElementsByTagName("answer")  # list of all senses
    train_data = {}
    for i in range(len(id_sense_list)):
        text_id = id_sense_list[i].getAttribute("instance")
        train_data[text_id] = {}
        train_data[text_id]['sense'] = id_sense_list[i].getAttribute("senseid")
        text_para = context_list[i].getElementsByTagName("s")
        train_data[text_id]['context'] = []
        for j in range(len(text_para)):
            target_sense = text_para[j].getElementsByTagName("head")
            if not target_sense:
                continue
            if target_sense[0].firstChild.data in ['line', 'lines']:  # convert all "lines" to "line" only for simplicity
                train_data[text_id]['context'].append(
                    text_para[j].childNodes[0].data + 'line' + text_para[j].childNodes[2].data)
            elif target_sense[0].firstChild.data == "Lines":
                train_data[text_id]['context'].append('line' + text_para[j].childNodes[1].data)
            else:
                train_data[text_id]['context'].append(text_para[j].childNodes[0].data + "line")
    return train_data


# Method to get training context ranked in order of the log likelihood
# Input: training data and k. k defines the context word to take, which will be +/- k from the target word "line"
# Return: a reverse ordered sorted list of tuples containing 3 values:
#           log_score = log likelihood value of the context
#           context = context word(s) taken based on Yarowsky algorithm
#           target_sense = the favored sense word for this context
def get_training_context(train_data, k):
    context_list = []
    sense_list = []
    for text_id in train_data:
        sense_context = train_data[text_id]
        for context in sense_context['context']:
            context = context.replace('lines', 'line')  # converting different variation of "line" for simplicity
            context = context.replace('Lines', 'line')
            context = context.replace('Line', 'line')
            word_token = word_tokenize(context)
            if "line" in word_token:  # taking only those context sentences that contains our target word
                context_list.append(word_token)
                sense_list.append(sense_context['sense'])
    senses = list(set(sense_list))  # all possible senses of our target words i.e. Phone and Product
    context_sense_score = {}
    for context in context_list:  # for each context sentences
        idx = context_list.index(context)  # get the id
        if "line" not in context:
            continue
        index = context.index("line")  # get the id for target word in sentence
        if index - 1 >= 0:  # 1st rule Position: -1 w
            target_context = " ".join(context[index - 1:index + 1])
            if target_context not in context_sense_score.keys():
                context_sense_score[target_context] = defaultdict(int)
            context_sense_score[target_context][sense_list[idx]] += 1
        if index + 1 < len(context):  # 2nd rule Position: +1 w
            target_context = " ".join(context[index:index + 2])
            if target_context not in context_sense_score.keys():
                context_sense_score[target_context] = defaultdict(int)
            context_sense_score[target_context][sense_list[idx]] += 1
        if index + 2 < len(context):  # 3rd rule Position: +1 w +2 w
            target_context = " ".join(context[index:index + 3])
            if target_context not in context_sense_score.keys():
                context_sense_score[target_context] = defaultdict(int)
            context_sense_score[target_context][sense_list[idx]] += 1
        if index - 2 > 0:  # 4th rule Position: -1 w -2 w
            target_context = " ".join(context[index - 2:index + 1])
            if target_context not in context_sense_score.keys():
                context_sense_score[target_context] = defaultdict(int)
            context_sense_score[target_context][sense_list[idx]] += 1
        if index - k > 0:  # 5th rule Position: -k w
            target_context = context[index - k] + "|line"
            if target_context not in context_sense_score.keys():
                context_sense_score[target_context] = defaultdict(int)
            context_sense_score[target_context][sense_list[idx]] += 1
        if index + k < len(context):  # 5th rule Position: +k w
            target_context = "line|"+context[index + k]
            if target_context not in context_sense_score.keys():
                context_sense_score[target_context] = defaultdict(int)
            context_sense_score[target_context][sense_list[idx]] += 1
    sorted_context_sense = get_log_likelihood(context_sense_score, senses)  # get the log likelihood score
    write_decision_list(sorted_context_sense)  # write the decision list for debugging
    return sorted_context_sense


# Method to get log likelihood score of each context based on how many times it appears for each sense
# Formulae: score_i = abs(log(P(sense_1|context_i)/P(sense_2)|context_i)))
# Input: context_sense_score dict and total sense list
# Return: a reverse ordered sorted list of tuples containing 3 values:
#           log_score = log likelihood value of the context
#           context = context word(s) taken based on Yarowsky algorithm
#           target_sense = the favored sense word for this context
def get_log_likelihood(context_sense_score, senses):
    sorted_context_sense = []
    sense1, sense2 = senses
    for context in context_sense_score.keys():
        target_sense = max(context_sense_score[context].items(), key=operator.itemgetter(1))[0]
        log_score = abs(math.log(max(0.1, context_sense_score[context][sense1])/max(0.1, context_sense_score[context][sense2])))
        sorted_context_sense.append((log_score, context, target_sense))
    sorted_context_sense = sorted(sorted_context_sense, key=lambda x: x[0], reverse=True)
    return sorted_context_sense


# Method to write the log likelihood score to my-decision-list.txt
# Input: log likelihood score tupele sorted_context_Sense
def write_decision_list(sorted_context_sense):
    with open("my-decision-list.txt", "w") as f:
        for log_score, context, target_sense in sorted_context_sense:
            f.write("%.3f\t%s\t%s\n" % (log_score, context, target_sense))
    return


# Method to get test data
# Input: file_path path of the testing file taken from arguments
# Return: A Python Dictionary test_data, which contains for each instance id,
#         all the context sentence where the target word appears
def get_test_data(file_path):
    DOMTree = xml.dom.minidom.parse(file_path)
    collection = DOMTree.documentElement
    instance_list = collection.getElementsByTagName("instance")
    context_list = collection.getElementsByTagName("context")
    test_data = {}
    for i in range(len(instance_list)):
        text_id = instance_list[i].getAttribute("id")
        test_data[text_id] = {}
        test_data[text_id]['sense'] = ""
        text_para = context_list[i].getElementsByTagName("s")
        test_data[text_id]['context'] = []
        for j in range(len(text_para)):
            target_sense = text_para[j].getElementsByTagName("head")
            if not target_sense:
                continue
            if target_sense[0].firstChild.data in ['line', 'lines']:
                test_data[text_id]['context'].append(
                    text_para[j].childNodes[0].data + 'line' + text_para[j].childNodes[2].data)
            elif target_sense[0].firstChild.data == "Lines":
                test_data[text_id]['context'].append('line' + text_para[j].childNodes[1].data)
            else:
                test_data[text_id]['context'].append(text_para[j].childNodes[0].data + "line")
    return test_data


# Method to get the most favored sense based on sorted_context_sense list
# Input: test_data, the sorted_context_sense and k
# Return: test_data with labeled senses
def get_test_sense(test_data, sorted_context_sense, k):
    for text_id in test_data.keys():
        for text_data in test_data[text_id]["context"]:
            text_data_tokens = word_tokenize(text_data)
            index = text_data_tokens.index("line")
            for log_score, context, target_sense in sorted_context_sense:
                if "line|" in context:  #+k word
                    kth_word = context.split("|")[-1]
                    if index+k < len(text_data_tokens):
                        if text_data_tokens[index+k] == kth_word:
                            test_data[text_id]["sense"] = target_sense
                            break
                elif "|line" in context:  # -k word
                    kth_word = context.split("|")[0]
                    if index - k >=0:
                        if text_data_tokens[index - k] == kth_word:
                            test_data[text_id]["sense"] = target_sense
                            break
                else:
                    if context in text_data:
                        # print("Mila!", context, " data: ", text_data)
                        test_data[text_id]["sense"] = target_sense
                        break
        if test_data[text_id]["sense"] == "":
            test_data[text_id]["sense"] = "phone"  # default case
    return test_data


# Method to print test_data id and sense
# Input: labeled test_data
def write_sense_output(test_data):
    for text_id in test_data.keys():
        sense = test_data[text_id]["sense"]
        print('<answer instance=' + "'" + text_id + "'" + " senseid=" + "'" + sense + "'/>")


# main function
def main():
    train_data = get_training_data(sys.argv[1])
    sorted_context_sense = get_training_context(train_data, 3)
    test_data = get_test_data(sys.argv[2])
    test_data_labeled = get_test_sense(test_data, sorted_context_sense, 3)
    write_sense_output(test_data_labeled)


if __name__ == '__main__':
    main()
