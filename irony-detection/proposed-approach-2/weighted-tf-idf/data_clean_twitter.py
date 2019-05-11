"""
AIT 690 Project Natural Language Processing
Team name : A team has no name
Members: Vidhyasri, Rahul Pandey, Arjun Mudumbi Srinivasan

This python script is used for cleaning document text. The method clean_str is used by all our approaches:
Some of the cleaning steps performed are:
    1. Expanding contraction dict (could've -> could have)
    2. Expand Hashtags (#IAmJoking -> i am joking)
    3. Replace all url by one token "_url_"
    4. &amp; -> &
    5. Replace all mentions with on token "_mention_"
    6. Adding spacing between punctuations and words
    7. Replace all numbers by spaces
    8. Replace all other special characters with spaces
    9. Remove all redundant spaces
"""
import re

FILE_NAME = ""

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
    # print(all_hashtags)
    # print()
    for ht in all_hashtags:
        string = string.replace(ht, expand_hash(ht))
    # string = emoji.demojize(string)
    string = string.lower()
    string = expand_contractions(string)
    string = re.sub(r'(http\S*)', '_URL_', string)
    string = re.sub(r'(https\S*)', '_URL_', string)
    # string = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '_URL_', string)
    string = re.sub(r'(&amp;)', '&', string)
    # string=re.sub(r'(RT)','',string)
    string = re.sub(r'(_NUM_)', '&', string)
    string = re.sub(r'(&quot;)', '"', string)
    string = re.sub(r'([0-9])', ' ', string)
    # d[0]=re.sub(r'([0-9])',' ',d[0])
    string = re.sub(r'(@\S*)', '_mention_', string)
    # string = re.sub(r'(@\S*)', '', string)
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

# test string for checking the cleaned file
# cln = "Lol! I m enjoying it👭 #Talks #Crazyness #SheNeverLeft  sleepy!! ☺"
# print(clean_str(cln))
