# Irony Detection on Online Social Media

### VidhyaSri, Rahul, and Arjun

Libraries needed:
Numpy, Pandas, scikitlearn,regex, nltk, gensim. tensorflow,sys, os, lightgbm,xgboost

Three Approach
1. Baseline:
    * subfolder: baseline-approach
    * uses BoW features on preprocessed text and trained on different classifiers

2. Proposed Approach 1:
    * subfolder proposed-approach-1
    * uses average of pretrained embeddings of each words as a feature of preprocessed document and trained on strong classifiers

3. Proposed Approach 2:
    1. Weighted tf-idf embeddings
        * subfolder weighted-tf-idf
        * uses weighted average of pretrained embeddings with tf-idf values as weights of each words as a feature of preprocessed document and trained on strong classifiers
    2. CNN for sentence classification
        * subfolder cnn
        * built on top of this github code [link](), which implements this paper [link]()
        * use train.py to train CNN, eval.py to get the fixed features, predictions on test data, and finally transfer cnn to get the final accuracy and f1 of different settings

Every file has comments on how to run them. Please read it before executing.

Data Fetching
1. SemEval Data for training and testing
    * [https://github.com/Cyvhee/SemEval2018-Task3](https://github.com/Cyvhee/SemEval2018-Task3)
    * In the datasets folder 
        * training file taskA: train/SemEval2018-T3-train-taskA_emoji.txt
        * training file taskB: train/SemEval2018-T3-train-taskB_emoji.txt
        * testing file taskA: goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt
        * testing file taskB: goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt

2. GloVe embeddings:
    * [http://nlp.stanford.edu/data/glove.twitter.27B.zip](http://nlp.stanford.edu/data/glove.twitter.27B.zip)

3. Word2Vec embeddings:
    * [https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)

4. Sampled 20000 predicted Reddit Comments:
    * [Link to Download](https://exchangelabsgmu-my.sharepoint.com/:x:/g/personal/rpandey4_masonlive_gmu_edu/EdBbIXasILxHgDJ-UGPcAOsBMBfn9qaHIUh4UDewCMiR_w?e=OkYJmW)

