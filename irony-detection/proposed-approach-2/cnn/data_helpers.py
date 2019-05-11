"""
AIT 690 Project Natural Language Processing
Team name : A team has no name
Members: Vidhyasri, Rahul Pandey, Arjun Mudumbi Srinivasan

This python script is built on top of this Github Code:
https://github.com/dennybritz/cnn-text-classification-tf
which implements Kim's CNN for sentence classification (https://arxiv.org/abs/1408.5882)

This python script is used as a building block for our proposed CNN method.

It has methods for:
    1. Cleaning document as per our previous approaches
    2. Loading data (get_dataset_semeval())
    3. Loading embeddings (load_embeddings...)
    4. Storing data in batches (batch_iter())

All these methods are used by train.py, eval.py
"""
import numpy as np
import re
import nltk
import random
import json
from sklearn.datasets import fetch_20newsgroups
from sklearn.datasets import load_files


AUX_LIST = ["am", "is", "are", "was", "were", "being", "been", "does", "do", "did", "has", "have", "had", "having", "can", "could", "may", "might", "must", "ough", "to", "shall", "should", "will", "would"]

contractions_dict = {
    "ain\'t": "am not",
    "aren\'t": "are not",
    "can\'t": "cannot",
    "can\'t\'ve": "cannot have",
    "\'cause": "because",
    "could\'ve": "could have",
    "couldn\'t": "could not",
    "couldn\'t\'ve": "could not have",
    "didn\'t": "did not",
    "doesn\'t": "does not",
    "don\'t": "do not",
    "hadn\'t": "had not",
    "hadn\'t\'ve": "had not have",
    "hasn\'t": "has not",
    "haven\'t": "have not",
    "he\'d": "he would",
    "he\'d\'ve": "he would have",
    "he\'ll": "he will",
    "he\'ll\'ve": "he will have",
    "he\'s": "he is",
    "how\'d": "how did",
    "how\'d\'y": "how do you",
    "how\'ll": "how will",
    "how\'s": "how is",
    "I\'d": "I would",
    "I\'d\'ve": "I would have",
    "I\'ll": "I will",
    "I\'ll\'ve": "I will have",
    "I\'m": "I am",
    "I\'ve": "I have",
    "isn\'t": "is not",
    "it\'d": "it would",
    "it\'d\'ve": "it would have",
    "it\'ll": "it will",
    "it\'ll\'ve": "it will have",
    "it\'s": "it is",
    "let\'s": "let us",
    "ma\'am": "madam",
    "mayn\'t": "may not",
    "might\'ve": "might have",
    "mightn\'t": "might not",
    "mightn\'t\'ve": "might not have",
    "must\'ve": "must have",
    "mustn\'t": "must not",
    "mustn\'t\'ve": "must not have",
    "needn\'t": "need not",
    "needn\'t\'ve": "need not have",
    "o\'clock": "of the clock",
    "oughtn\'t": "ought not",
    "oughtn\'t\'ve": "ought not have",
    "shan\'t": "shall not",
    "sha\'n\'t": "shall not",
    "shan\'t\'ve": "shall not have",
    "she\'d": "she she would",
    "she\'d\'ve": "she would have",
    "she\'ll": "she will",
    "she\'ll\'ve": "she will have",
    "she\'s": "she is",
    "should\'ve": "should have",
    "shouldn\'t": "should not",
    "shouldn\'t\'ve": "should not have",
    "so\'ve": "so have",
    "so\'s": "so is",
    "that\'d": "that had",
    "that\'d\'ve": "that would have",
    "that\'s": "that is",
    "there\'d": "there would",
    "there\'d\'ve": "there would have",
    "there\'s": "there is",
    "they\'d": "they would",
    "they\'d\'ve": "they would have",
    "they\'ll": "they will",
    "they\'ll\'ve": "they will have",
    "they\'re": "they are",
    "they\'ve": "they have",
    "to\'ve": "to have",
    "wasn\'t": "was not",
    "we\'d": "we would",
    "we\'d\'ve": "we would have",
    "we\'ll": "we will",
    "we\'ll\'ve": "we will have",
    "we\'re": "we are",
    "we\'ve": "we have",
    "weren\'t": "were not",
    "what\'ll": "what will",
    "what\'ll\'ve": "what will have",
    "what\'re": "what are",
    "what\'s": "what is",
    "what\'ve": "what have",
    "when\'s": "when is",
    "when\'ve": "when have",
    "where\'d": "where did",
    "where\'s": "where is",
    "where\'ve": "where have",
    "who\'ll": "who will",
    "who\'ll\'ve": "who will have",
    "who\'s": "who is",
    "who\'ve": "who have",
    "why\'s": "why is",
    "why\'ve": "why have",
    "will\'ve": "will have",
    "won\'t": "will not",
    "won\'t\'ve": "will not have",
    "would\'ve": "would have",
    "wouldn\'t": "would not",
    "wouldn\'t\'ve": "would not have",
    "y\'all": "you all",
    "y\'all\'d": "you all would",
    "y\'all\'d\'ve": "you all would have",
    "y\'all\'re": "you all are",
    "y\'all\'ve": "you all have",
    "you\'d": "you would",
    "you\'d\'ve": "you would have",
    "you\'ll": "you will",
    "you\'ll\'ve": "you shall have",
    "you\'re": "you are",
    "you\'ve": "you have"
}


contractions_re = re.compile('(%s)' % '|'.join(contractions_dict.keys()))


def expand_contractions(s, contractions_dict=contractions_dict):
    def replace(match):
        return contractions_dict[match.group(0)]
    return contractions_re.sub(replace, s)


expand_hash = lambda i: " ".join([a for a in re.split('([A-Z][a-z]+)', i) if a])


def clean_str(string):
    all_hashtags = re.findall(r'#(\w+)', string)
    for ht in all_hashtags:
        string = string.replace(ht, expand_hash(ht))
    string = string.lower()
    string = expand_contractions(string)
    string=re.sub(r'(http\S*)','_URL_',string)
    string=re.sub(r'(https\S*)','_URL_',string)
    string=re.sub(r'(&amp;)','&',string)
    # string=re.sub(r'(RT)','',string)
    string=re.sub(r'(_NUM_)','&',string)
    string=re.sub(r'(&quot;)','"',string)
    string=re.sub(r'([0-9])',' ',string)
    string=re.sub(r'(@\S*)','_mention_',string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(' +', ' ', string)
    string = string.strip().lower()
    return string


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def get_datasets_20newsgroup(subset='train', categories=None, shuffle=True, random_state=42):
    """
    Retrieve data from 20 newsgroups
    :param subset: train, test or all
    :param categories: List of newsgroup name
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the newsgroup
    """
    datasets = fetch_20newsgroups(subset=subset, categories=categories, shuffle=shuffle, random_state=random_state)
    return datasets


def get_datasets_semeval(path):
    data_label = []
    datasets = {'data': [], 'target': [], 'target_names': []}
    i = 0
    # train_path = '/Users/rahulpandey/Downloads/SemEval2018-Task3-master/datasets/train/SemEval2018-T3-train-taskA_emoji.txt'
    for lines in open(path):
        if i==0:
            i = 1
            continue
        lines = lines.strip()
        lines = lines.split('\t')
        if lines[1] not in datasets['target_names']:
            datasets['target_names'].append(lines[1])
        data_label.append((lines[2], datasets['target_names'].index(lines[1])))

    random.shuffle(data_label)
    random.shuffle(data_label)
    for item in data_label:
        datasets['data'].append(item[0])
        datasets['target'].append(item[1])
    return datasets


def get_datasets_mrpolarity(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]

    datasets = dict()
    datasets['data'] = positive_examples + negative_examples
    target = [0 for x in positive_examples] + [1 for x in negative_examples]
    datasets['target'] = target
    datasets['target_names'] = ['positive_examples', 'negative_examples']
    return datasets


def get_datasets_localdata(container_path=None, categories=None, load_content=True,
                       encoding='utf-8', shuffle=True, random_state=42):
    """
    Load text files with categories as subfolder names.
    Individual samples are assumed to be files stored a two levels folder structure.
    :param container_path: The path of the container
    :param categories: List of classes to choose, all classes are chosen by default (if empty or omitted)
    :param shuffle: shuffle the list or not
    :param random_state: seed integer to shuffle the dataset
    :return: data and labels of the dataset
    """
    datasets = load_files(container_path=container_path, categories=categories,
                          load_content=load_content, shuffle=shuffle, encoding=encoding,
                          random_state=random_state)
    return datasets


def load_data_labels(datasets):
    """
    Load data and labels
    :param datasets:
    :return:
    """
    # Split by words
    x_text = datasets['data']
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    labels = []
    for i in range(len(x_text)):
        label = [0 for j in datasets['target_names']]
        label[datasets['target'][i]] = 1
        labels.append(label)
    y = np.array(labels)
    return [x_text, y]


def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors
