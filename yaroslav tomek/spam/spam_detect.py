import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import PassiveAggressiveClassifier


# specify location your dataset here
DATA_PATH = r"C:\Users\User\Documents\Git\yaroslav tomek\data\dataset.txt"

# give name to label-column and text-column
COLUMN_LABEL = "label"
COLUMN_TEXT = "text"

# these are labels that indicate the type of message.
LABEL_LEGIT = 'LEGI'
LABEL_SPAM = 'SPAM'
LABEL_SMISHING = 'SMIS'

dataset = pd.read_csv(DATA_PATH, sep='\t', names=[COLUMN_LABEL, COLUMN_TEXT], header=None)
print('Total size:', dataset.shape[0])
print('Legit messages:', dataset[dataset[COLUMN_LABEL] == LABEL_LEGIT].shape[0])
print('Spam messages:', dataset[dataset[COLUMN_LABEL] == LABEL_SPAM].shape[0])
print('Smishing messages:', dataset[dataset[COLUMN_LABEL] == LABEL_SMISHING].shape[0])

dataset = dataset[((dataset[COLUMN_LABEL] == LABEL_LEGIT) | (dataset[COLUMN_LABEL] == LABEL_SPAM))]
# Let's check if they are gone
#print('Smishing messages:', dataset[dataset[COLUMN_LABEL] == LABEL_SMISHING].shape[0])

def messages2vectors(messages):
    '''
    Transforms single message into feature-vector;
    Parameters:
        messages    -   array of strings;
    Returns:
        features    -   array of feature-vectors;
    '''

    elmo = hub.Module("https://tfhub.dev/google/elmo/1")

    features = np.zeros((0, 1024))
    n = 100
    l = int(len(messages) / n) if len(messages) % n == 0 else int(len(messages) / n) + 1
    for i in range(l):

        if (i + 1) * n < len(messages):
            right = (i + 1) * n
            embedds = elmo(messages[int(i * n) : right], signature="default", as_dict=True)["default"]
        else:
            embedds = elmo(messages[:len(messages) - int(i * n)], signature="default", as_dict=True)["default"]

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            embedds = sess.run(embedds)
            features = np.concatenate([features, embedds])

    return features

def convert_labels(labels_raw):
    '''
    Transforms labels into numerical values;
    Parameters:
        labels_raw    -   array of text-labels;
    Returns:
        features    -   array of numerical labels;
    '''

    labels = labels_raw.replace('LEGI', 0)
    labels = labels.replace('SPAM', 1)

    return labels

features = messages2vectors(dataset[COLUMN_TEXT])
labels = convert_labels(dataset[COLUMN_LABEL])
#print(features)
print(features.shape)
print(labels.shape)
#print(labels)

def split_data(features, labels, ratio=0.7):
    '''
    Splits dataset into train/test parts using given ratio;
    Parameters:
        data    -   array of features;
        labels  -   array of corresponding labels;
        ratio   -   train/test size ratio;
    Returns:
        train_data      -   array of training features;
        train_labels    -   array of training labels;
        test_data       -   array of testing features;
        test_labels     -   array of testing labels;
    '''

    positive_data = features[labels == 1]  # all spam features
    negative_data = features[labels == 0]  # all legit features

    # We shuffle arrays to get random samples later
    random_indecies_positive = np.arange(positive_data.shape[0])
    np.random.shuffle(random_indecies_positive)
    random_indecies_negative = np.arange(negative_data.shape[0])
    np.random.shuffle(random_indecies_negative)

    n_positive_train = int(positive_data.shape[0] * ratio)
    n_negative_train = int(negative_data.shape[0] * ratio)

    # Training data are all indecies in 'ratio' part of shuffled indecies
    train_data = np.concatenate([positive_data[random_indecies_positive[:n_positive_train]],
                                 negative_data[random_indecies_negative[:n_negative_train]]])

    train_labels = np.asarray([1] * n_positive_train + [0] * n_negative_train)

    # Testing data are all indecies that remain
    test_data = np.concatenate([positive_data[random_indecies_positive[n_positive_train:]],
                                negative_data[random_indecies_negative[n_negative_train:]]])

    test_labels = np.asarray(
        [1] * (positive_data.shape[0] - n_positive_train) + [0] * (negative_data.shape[0] - n_negative_train))

    return train_data, train_labels, test_data, test_labels

#print(split_data(features,labels))

def get_metrics(labels, predictions):
    '''
    Computes metrics;
    Parameters:
        labels    -   array of labels;
        predictions  -   array of predictions;
    Returns:
        FAR -   False Acceptance Rate;
        FRR -   False Rejection Rate;
    '''
    #confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    print(pd.DataFrame(confusion_matrix(labels, predictions),
                       columns=['Predicted Spam', "Predicted Legi"], index=['Actual Spam', 'Actual Legi']))
    '''
    print(f'\nTrue Positives: {tp}')
    print(f'False Positives: {fp}')
    print(f'True Negatives: {tn}')
    print(f'False Negatives: {fn}')
    '''
    FAR = fn/(fn+tp)
    FRR = fp/(fp+tn)
    return FAR, FRR

def evaluate(classifierType, hyperparameters, features, labels):
    '''
    Splits dataset into train/test parts using given ratio;
    Parameters:
        classifierType      -   type of ML algorithm to use;
        hyperparameters     -   dictionary of model's parameters;
        features            -   array of features;
        labels              -   array of labels
    Returns:
        trainFAR    -   False Acceptance Rate for train dataset;
        trainFRR    -   False Rejection Rate for train dataset;
        testFAR     -   False Acceptance Rate for test dataset;
        testFRR    -   False Rejection Rate for test dataset;
    '''

    model = classifierType(**hyperparameters)

    # Split data
    train_data, train_labels, test_data, test_labels = split_data(features, labels)

    print('Train set shape:', train_data.shape)
    print('Train labels shape:', train_labels.shape)
    print('Test set shape:', test_data.shape)
    print('Test labels shape:', test_labels.shape)

    # Fit your model
    model=model.fit(train_data, train_labels)

    # Make predictions for training dataset
    print("---TRAINING---")
    prediction_train=model.predict(train_data)

    # Compute train FAR/FRR
    trainFAR, trainFRR = get_metrics(train_labels, prediction_train)

    # Make predictions for testing dataset
    predictions_test = model.predict(test_data)

    # Compute test FAR/FRR
    print("---TESTING---")
    testFAR, testFRR = get_metrics(test_labels, predictions_test)

    return trainFAR, trainFRR, testFAR, testFRR

classifierType1 = sklearn.ensemble.RandomForestClassifier
hyperparameters1 = {'n_estimators' : 600,
                'criterion' : 'entropy',
                'max_depth': 8,
                'min_samples_split' : 3,
                'min_samples_leaf': 1,
                'min_weight_fraction_leaf': 0.0,
                'max_features': 'auto',
                'max_leaf_nodes': None,
                'min_impurity_decrease': 0}
print("--RANDOM FOREST CLASSIFIER--")
print(evaluate(classifierType1, hyperparameters1, features, labels))

classifierType2 = sklearn.naive_bayes.BernoulliNB
hyperparameters2 = {'alpha':0.0,
                'binarize':0.0,
                'fit_prior':False,
                'class_prior':None,
                'fit_prior': True}
print("--NATIVE BAYES BERNOULLI--")
print(evaluate(classifierType2, hyperparameters2, features, labels))

classifierType3 = sklearn.linear_model.PassiveAggressiveClassifier
hyperparameters3 = {'C':0.5,
                    'fit_intercept':True,
                    'max_iter':1000,
                    'tol':0.0001,
                    'early_stopping':False,
                    'n_iter_no_change':3,
                    'shuffle':True,
                    'verbose':2,
                    'loss':'hinge',
                    'n_jobs':None,
                    'random_state':None,
                    'warm_start':False,
                    'class_weight':None,
                    'average':False}

print("--PASSIVEAGRESSIVE CLASSIFIER--")
print(evaluate(classifierType3, hyperparameters3, features, labels))
