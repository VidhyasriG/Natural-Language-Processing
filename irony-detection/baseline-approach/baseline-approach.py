"""
AIT 690 Project Natural Language Processing
Team name : A team has no name
Members: Vidhyasri, Rahul Pandey, Arjun Mudumbi Srinivasan

This python script is used to develop the base model of the sarcasm/irony detection. We start with importing the necessary libraries , reading the data
from the text file, followed by splitting the data into columns and making a dataframe. The data is cleaned using few functions which involves removing the mentions
urls and others . Bag of words model is developed using the Sklearn's TFIDF vectorizer.
data is split into train and test and the models like svm , decision tree is used to predict. The accuracy and F-1 scores are determined using sklearn implementation

File Usage :
$ python baseline-approach.py training_file_path test_file_path
where:
	training_file_path: Can be SemEval2018-T3-train-taskA_emoji.txt or SemEval2018-T3-train-taskB_emoji.txt. Depending on whether task A or task B is being performed
	test_file_path: Can be SemEval2018-T3-test-taskA_emoji.txt or SemEval2018-T3-test-taskB_emoji.txt. Depending on whether task A or task B is being performed

Output: Accuracy and F-1 scores of different models and confusion matrix of the best performing baseline model i.e. SVM
"""
# Importing the necessary libraries
import pandas as pd
import sys
from data_clean_twitter import *
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


ct = 0  # Since the first column is the name of the column in the file, we use count to ignore it
serial_number = re.compile(r'^([0-9]+)')  # regex to extract the serial number
label_num = re.compile(r'\t([0-1])\t')  #  regex to extract the label
text_reg = re.compile(r'\t[0-1]\t(.+)$')  #  regex to extract the text
snum = []  #  array to dump 
label = []
text = []
with open(sys.argv[1], "r") as file:  # opent the train file
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
data["text"] = data["text"].apply(clean_str)  #  apply the cleaning function imported from data_clean_twitter# The code to remove stopwords and data cleaning has been commented below# ps = WordNetLemmatizer()# stop_words = set(stopwords.words('english'))#  stop_words = set(stop_words)#  data["filtered_sentence"] = pd.Series()#  for i in range(len(data)):#  	text = str(data["text"].iloc[i])#  	stem_text = ""#  	word_tokens = re.split('\W+', text)#  	filtered_sentence = " ".join([w for w in word_tokens if not w in stop_words])#  	text = "".join([w for w in str(filtered_sentence) if w not in string.punctuation])#  	word_tokens = re.split('\W+', text)#  	stem_text = " ".join((ps.lemmatize(str(w)) for w in word_tokens))#  	data["filtered_sentence"].iloc[i] = stem_text# word2vec_tokenize = word_tokenize(p["filtered_sentence"].iloc[i])
vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features=200)  # generate a TFIDF vectorizer with 200 features and  unigrams only
data2 = vectorizer.fit_transform(str(data.loc[i, "text"]) for i in range(len(data)))  # fit and transform the tfidf vectorizer to the data 
data2 = data2.toarray()  # Convert the data to an numpy array#  data2 = pd.DataFrame(data2.toarray(), columns = vectorizer.get_feature_names())# print(len(vectorizer.get_feature_names()))#  from sklearn.feature_selection import SelectKBest, chi2
feature_names = vectorizer.get_feature_names()  # get the feature names from the TFIDF
y = data["label"]  # generate the labe; data# %%%%%%%% Import all the libraries for the model
# create estimators using all the classifier
estimator_nb = GaussianNB()
estimator_svc = SVC(kernel = "linear")
estimator_lr = LogisticRegression()
estimator_knn = KNeighborsClassifier()
estimator_dt = DecisionTreeClassifier()
print("Now training")
estimator_svc.fit(data2, y)
estimator_lr.fit(data2, y)
estimator_nb.fit(data2, y)
estimator_knn.fit(data2, y)
estimator_dt.fit(data2, y)
print("Training done")  # %%%%%%%%%%%%%%%%%%%%%%%%# the test dataset has been read and similar preprocessing as the training data is performed
ct = 0
serial_number  = re.compile(r'^([0-9]+)')
label_num = re.compile(r'\t([0-1])\t')
text_reg = re.compile(r'\t[0-1]\t(.+)$')
snum = []
label = []
text = []  # base_path_test = "/home/arjun/Desktop/Fall-2018/AIT NLP/Assignments/Project/ait-690-projects-master/datasets/goldtest_TaskA"
with open(sys.argv[2], "r") as file:  # open thes test file
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
x_pred = vectorizer.transform(str(data_test.loc[i, "text"]) for i in range(len(data_test))).toarray()# Now predicting the test data
y_pred_svc = estimator_svc.predict(x_pred)
y_pred_lr = estimator_lr.predict(x_pred)
y_pred_nb = estimator_nb.predict(x_pred)
y_pred_knn = estimator_knn.predict(x_pred)
y_pred_dt = estimator_dt.predict(x_pred)  # print(y_true, y_pred)
print("Model:%s|Accuracy: %.3f %% | F1: %.3f %%" % ("[SVM] Support Vector Classifier", 100*accuracy_score(y_true, y_pred_svc), 100*f1_score(y_true, y_pred_svc)))
print("Model:%s|Accuracy: %.3f %% | F1: %.3f %%" % ("[LR] Logistic Regression", 100*accuracy_score(y_true, y_pred_lr), 100*f1_score(y_true, y_pred_lr)))
print("Model:%s|Accuracy: %.3f %% | F1: %.3f %%" % ("[KNN] K Nearest Neigbhors", 100*accuracy_score(y_true, y_pred_knn), 100*f1_score(y_true, y_pred_knn)))
print("Model:%s|Accuracy: %.3f %% | F1: %.3f %%" % ("[DT] Decision Tree", 100*accuracy_score(y_true, y_pred_dt), 100*f1_score(y_true, y_pred_dt)))
print("Model:%s|Accuracy: %.3f %% | F1: %.3f %%" % ("[NB] Naive Bayes", 100*accuracy_score(y_true, y_pred_nb), 100*f1_score(y_true, y_pred_nb)))
print("Confusion matrix of the best classifier (SVM)")
print(confusion_matrix(y_true, y_pred_svc))