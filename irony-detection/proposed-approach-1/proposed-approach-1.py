"""
AIT 690 Project Natural Language Processing
Team name : A team has no name
Members: Vidhyasri, Rahul Pandey, Arjun Mudumbi Srinivasan

This python script is used to develop the first proposed approach model of the sarcasm/irony detection. We start with importing the necessary libraries , reading the data
from the text file, followed by splitting the data into columns and making a dataframe. The data is cleaned using few new functions compared to base model which involves removing the mentions
urls and others, expanding hashtags(#IAmJoking -> i am joking), contraction words ("There's -> there is"), etc..

We created the average of word embeddings as features. We used GloVe vectors and Google's word2vec as embedding features
Data is split into train and test and the models like svm , decision tree is used to predict. The accuracy and F-1 scores are determined using sklearn implementation

File Usage :
$ python proposed-approach-1.py training_file_path test_file_path embedding_type embedding_path
where:
	training_file_path: Can be SemEval2018-T3-train-taskA_emoji.txt or SemEval2018-T3-train-taskB_emoji.txt. Depending on whether task A or task B is being performed
	test_file_path: Can be SemEval2018-T3-test-taskA_emoji.txt or SemEval2018-T3-test-taskB_emoji.txt. Depending on whether task A or task B is being performed
	embedding_type: glove/word2vec any of the one
	embedding_path: File path of the embedding file (GloVe/Word2vec)

Output: Accuracy and F-1 scores of different models and confusion matrix of the best performing baseline model
"""
# Importing the necessary libraries
import pandas as pd
import numpy as np
import sys
from data_clean_twitter import *
import gensim
from sklearn.svm import SVR, SVC
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from nltk.tokenize import word_tokenize


# Method to load GloVe vectors
def get_model_glove():
	MODEL_PATH = sys.argv[4]  # get the GloVe Model Path
	vect_size = int(MODEL_PATH.split("/")[-1].split(".")[-2][:-1])
	print("Importing", MODEL_PATH)
	model = {}
	for lines in open(MODEL_PATH):
		lines = lines.rstrip()
		lines = lines.split(" ")
		model[lines[0]] = [float(x) for x in lines[1:]]
	return model, vect_size


# Method to load Google's word2vec vectors
def get_model_word2vec():
	MODEL_PATH = sys.argv[4]
	print("Importing", MODEL_PATH)
	model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=True)
	vect_size = 300  # default word2vec size
	return model, vect_size


# Method to create the average of word vectors of GloVe embeddings as document vectors
def get_word2vec_vectors(data, model, vect_size):
	X = []
	for i in range(len(data)):
		word_tokens = word_tokenize(data.loc[i, "text"])
		vects = np.zeros(vect_size)
		cnt_tkn = 0
		for word in word_tokens:
			if word in model.wv.vocab:
				cnt_tkn += 1
				vects += model.wv.word_vec(word)
		if cnt_tkn != 0:
			vects /= cnt_tkn
		X.append(vects)
	return X


# Method to create the average of word vectors of word2vec embeddings as document vectors
def get_glove_vectors(data, model, vect_size):
	X = []
	for i in range(len(data)):
		word_tokens = word_tokenize(data.loc[i, "text"])
		vects = np.zeros(vect_size)
		cnt_tkn = 0
		for word in word_tokens:
			if word in model.keys():
				cnt_tkn += 1
				vects += np.array(model[word])
		if cnt_tkn != 0:
			vects /= cnt_tkn
		X.append(vects)
	return X


ct = 0  # Since the first column is the name of the column in the file, we use count to ignore it
serial_number = re.compile(r'^([0-9]+)')  # regex to extract the serial number
label_num = re.compile(r'\t([0-1])\t')  #  regex to extract the label
text_reg = re.compile(r'\t[0-1]\t(.+)$')  #  regex to extract the text
snum = []  #  array to dump
label = []
text = []
with open(sys.argv[1], "r") as file:  # open the train file
	for l in file:
		if ct == 0:
			ct += 1
			continue
		else:
			p = serial_number.findall(str(l))  # find all of the necessary values
			snum.append(p[0])
			p = label_num.findall(str(l))
			label.append(int(p[0]))
			p = text_reg.findall(str(l))
			text.append(p[0])# print(text)
data = pd.DataFrame()  # create a dataframe and dump the values in to them
snum = pd.Series(snum)  # covert the list into pandas Series so as to dump into the dataframe
label = pd.Series(label)
text = pd.Series(text)
text.apply(clean_str)
data["snum"] = snum
data["label"] = label
data["text"] = text
data["text"] = data["text"].apply(clean_str)  # apply the cleaning function imported from data_clean_twitter# The code to remove stopwords and data cleaning has been commented below# ps = WordNetLemmatizer()# stop_words = set(stopwords.words('english'))#  stop_words = set(stop_words)#  data["filtered_sentence"] = pd.Series()#  for i in range(len(data)):#  	text = str(data["text"].iloc[i])#  	stem_text = ""#  	word_tokens = re.split('\W+', text)#  	filtered_sentence = " ".join([w for w in word_tokens if not w in stop_words])#  	text = "".join([w for w in str(filtered_sentence) if w not in string.punctuation])#  	word_tokens = re.split('\W+', text)#  	stem_text = " ".join((ps.lemmatize(str(w)) for w in word_tokens))#  	data["filtered_sentence"].iloc[i] = stem_text# word2vec_tokenize = word_tokenize(p["filtered_sentence"].iloc[i])

word_embedding_type = sys.argv[3]  # Check type of word embeddings GloVe or word2vec
if word_embedding_type == "glove":
	model, vect_size = get_model_glove()  # Get the GloVE model and vect_size of the embeddings
	X = get_glove_vectors(data, model, vect_size)  # Get the document vectors of training data
else:
	model, vect_size = get_model_word2vec()  # Get the word2vec model and vect_size of the embeddings
	X = get_word2vec_vectors(data, model, vect_size)  # Get the document vectors of training data
y = data["label"]  # generate the label data
# create estimators using all the classifier
estimator_svc = SVC(kernel="linear")
estimator_lr = LogisticRegression()
estimator_gbm = GradientBoostingClassifier()
estimator_lgbm = lgb.LGBMClassifier()
print("Now training")
estimator_svc.fit(X, y)
estimator_lr.fit(X, y)
estimator_gbm.fit(X, y)
estimator_lgbm.fit(X, y)
print("Training done")  # %%%%%%%%%%%%%%%%%%%%%%%%# the test dataset has been read and similar preprocessing as the training data is performed
ct = 0
serial_number = re.compile(r'^([0-9]+)')
label_num = re.compile(r'\t([0-1])\t')
text_reg = re.compile(r'\t[0-1]\t(.+)$')
snum = []
label = []
text = []
with open(sys.argv[2], "r") as file:  # open the test file
	for l in file:
		if ct == 0:
			ct += 1
			continue
		else:
			p = serial_number.findall(str(l))  # extract all the necessary values
			snum.append(p[0])
			p = label_num.findall(str(l))
			label.append(int(p[0]))
			p = text_reg.findall(str(l))
			text.append(p[0])

data_test = pd.DataFrame()  # create the dataframe
snum = pd.Series(snum)
label = pd.Series(label)  # dump the values using into the dataframe
text = pd.Series(text)
text.apply(clean_str)
data_test["snum"] = snum
data_test["label"] = label
data_test["text"] = text  # %%%%%%%%%%%%5
data_test["text"] = data_test["text"].apply(clean_str)  # clean the text file
y_true = data_test["label"].tolist()
if word_embedding_type == "glove":
	x_pred = get_glove_vectors(data_test, model, vect_size)
else:
	x_pred = get_word2vec_vectors(data_test, model, vect_size)
y_pred_svc = estimator_svc.predict(x_pred)
y_pred_lr = estimator_lr.predict(x_pred)
y_pred_gbm = estimator_gbm.predict(x_pred)
y_pred_lgbm = estimator_lgbm.predict(x_pred)
# store accuracy score of different estimators
accuracy_score_dict = {}
accuracy_score_dict["svc"] = 100 * accuracy_score(y_true, y_pred_svc)
accuracy_score_dict["lr"] = 100 * accuracy_score(y_true, y_pred_lr)
accuracy_score_dict["gbm"] = 100 * accuracy_score(y_true, y_pred_gbm)
accuracy_score_dict["lgbm"] = 100 * accuracy_score(y_true, y_pred_lgbm)
# store f1 score of different estimators
f1_score_dict = {}
f1_score_dict["svc"] = 100*f1_score(y_true, y_pred_svc)
f1_score_dict["lr"] = 100*f1_score(y_true, y_pred_lr)
f1_score_dict["gbm"] = 100*f1_score(y_true, y_pred_gbm)
f1_score_dict["lgbm"] = 100*f1_score(y_true, y_pred_lgbm)
# print(y_true, y_pred)
print("Model:%s|Accuracy: %.3f %% | F1: %.3f %%" % ("[SVM] Support Vector Classifier", accuracy_score_dict["svc"], f1_score_dict["svc"]))
print("Model:%s|Accuracy: %.3f %% | F1: %.3f %%" % ("[LR] Logistic Regression", accuracy_score_dict["lr"], f1_score_dict["lr"]))
print("Model:%s|Accuracy: %.3f %% | F1: %.3f %%" % ("[GBM] Gradient Boosting Model", accuracy_score_dict["gbm"], f1_score_dict["gbm"]))
print("Model:%s|Accuracy: %.3f %% | F1: %.3f %%" % ("[LGBM] Microsoft Light Gradient Boosting Model", accuracy_score_dict["lgbm"], f1_score_dict["lgbm"]))
if word_embedding_type == "glove":
	print("Confusion matrix of the best classifier (LR)")
	print(confusion_matrix(y_true, y_pred_lr))
else:
	print("Confusion matrix of the best classifier (LGBM)")
	print(confusion_matrix(y_true, y_pred_lgbm))