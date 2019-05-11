"""
AIT 690 Project Natural Language Processing
Team name : A team has no name
Members: Vidhyasri, Rahul Pandey, Arjun Mudumbi Srinivasan

This python script is built on top of this Github Code:
https://github.com/dennybritz/cnn-text-classification-tf
which implements Kim's CNN for sentence classification (https://arxiv.org/abs/1408.5882)

This code is used to evaluate the trained CNN model on test data and store information like:
    1. The last layer of each training document as an fixed input features
    2. The true label of test data
    3. The predicted label of test data

It also uses data_helper.py file to clean and load the data and also split the data into batches
The config.yml file contains information about word2vec path, glove path, etc.

File Usage :
$ python eval.py run_number checkpoint_number test_file_path data_type

Note: Please update the config.yml for local path of the embeddings

where:
	run_number: the triained weights folder name inside runs folder
	checkpoint_number: can be "latest" for latest checkpoint or the exact checkpoint number
	test_file_path: Can be SemEval2018-T3-test-taskA_emoji.txt or SemEval2018-T3-test-taskB_emoji.txt. Depending on whether task A or task B is being performed
	data_type: train/test it will be used to save the output file of different data type used for evaluation

Output: Pickle file containing information like last layer, true label, and predicted label by CNN stored in the training
        weights folder inside runs directory for different data (test or train)
"""
import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
import pickle
import yaml
import sys


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# Parameters
# ==================================================

# Data Parameters

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/"+sys.argv[1]+"/checkpoints/", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

datasets = None

# CHANGE THIS: Load data. Load your own data here
dataset_name = cfg["datasets"]["default"]
if FLAGS.eval_train:
    if dataset_name == "mrpolarity":
        datasets = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                             cfg["datasets"][dataset_name]["negative_data_file"]["path"])
    elif dataset_name == "20newsgroup":
        datasets = data_helpers.get_datasets_20newsgroup(subset="test",
                                              categories=cfg["datasets"][dataset_name]["categories"],
                                              shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                              random_state=cfg["datasets"][dataset_name]["random_state"])
    elif dataset_name == "semeval":
        test_path = sys.argv[3]
        datasets = data_helpers.get_datasets_semeval(test_path)
    x_raw, y_test = data_helpers.load_data_labels(datasets)
    y_test = np.argmax(y_test, axis=1)
    print("Total number of test examples: {}".format(len(y_test)))
else:
    if dataset_name == "mrpolarity":
        datasets = {"target_names": ['positive_examples', 'negative_examples']}
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]
    else:
        datasets = {"target_names": ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']}
        x_raw = ["The number of reported cases of gonorrhea in Colorado increased",
                 "I am in the market for a 24-bit graphics card for a PC"]
        y_test = [2, 1]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
# print("Checkpoint", FLAGS.checkpoint_dir)
checkpoint_number = sys.argv[2]
if checkpoint_number == "latest":
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
else:
    checkpoint_file = "runs/"+sys.argv[1]+"/checkpoints/model-"+checkpoint_number
print("check", checkpoint_file)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        scores = graph.get_operation_by_name("output/scores").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Fixed features output
        for op in graph.get_operations():
            str_op = str(op.name)
            if "fixed" in str_op:
                print (str_op)
        fixed_feat = graph.get_operation_by_name("fixed_feature").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []
        all_probabilities = None
        all_feat = None

        for x_test_batch in batches:
            batch_predictions_scores = sess.run([predictions, scores, fixed_feat], {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions_scores[0]])
            probabilities = softmax(batch_predictions_scores[1])
            feat_vectors = batch_predictions_scores[2]
            print("Feature vectors size", feat_vectors.shape, "probabilities size", probabilities.shape)
            # sys.exit()
            if all_probabilities is not None:
                all_probabilities = np.concatenate([all_probabilities, probabilities])
                all_feat = np.concatenate([all_feat, feat_vectors])
            else:
                all_probabilities = probabilities
                all_feat = feat_vectors

# Save the evaluation to a csv
# create the results file containing all necessary information
result = []
for i in range(len(y_test)):
    tmp_res = {}
    tmp_res["TEXT"] = x_raw[i]
    tmp_res["LABEL"] = y_test[i]
    tmp_res["PREDICTION"] = int(all_predictions[i])
    tmp_res["PROBABILITY"] = all_probabilities[i]
    tmp_res["FEATURES"] = all_feat[i]
    result.append(tmp_res)


out_path = os.path.join(FLAGS.checkpoint_dir, "..", "semeval_"+sys.argv[4]+"_prediction_last_layer_full.pkl")
print("Saving evaluation to {0}".format(out_path))
with open(out_path, 'wb') as f:
    pickle.dump(result, f)
