import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


# specify location your dataset here
DATA_PATH = "data.txt"

# give name to label-column and text-column
COLUMN_LABEL = "class"
COLUMN_TEXT = "context"

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
print('Smishing messages:', dataset[dataset[COLUMN_LABEL] == LABEL_SMISHING].shape[0])



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


    # add your code here
    labels = np.array([(0 if i=="LEGI" else 1) for i in labels_raw ])

    return labels

features = messages2vectors(dataset[COLUMN_TEXT])
labels = convert_labels(dataset[COLUMN_LABEL])
print(features.shape)
print(labels.shape)
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


    positive_data = features[labels == 1] # all spam features
    negative_data = features[labels == 0] # all legit features

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

    test_labels = np.asarray([1] * (positive_data.shape[0]  - n_positive_train) + [0] * (negative_data.shape[0] - n_negative_train))

    return train_data, train_labels, test_data, test_labels



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
    # add your code here


    cf = confusion_matrix(labels, predictions)
    FAR = cf[0][1]/(cf[0][1] + cf[0][0])
    FRR = cf[1][0]/(cf[1][0] + cf[1][1])
    return FAR, FRR




classifierType = [RandomForestClassifier, MultinomialNB,SVC]
hp = {'n_estimators' : 70,
                'criterion' : 'gini',
                'max_depth' : None,
                'min_samples_split' : 2,
                'n_jobs' :-1}



hyperparameters = {'n_estimators' : list(range(75,200,5)),
                'criterion' : ['gini'],
                'max_depth' : [None],
                'min_samples_split' : [1,2,3,5]
                }
            
'''
#after a few sugestions with gridsearch e.g {'criterion': 'gini', 'max_depth': None, 'min_samples_split': 5, 'n_estimators': 95}

hyperparameters = {'n_estimators' : [96,98,100,105,120,114,135,111,95],
                'criterion' : ['gini'],
                'max_depth' : [None],
                'min_samples_split' : [5]
                }
'''
def evaluate(classifierType, hyperparameters, features, labels,hp , cl = "not NB"):
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

   

    # Split data
    # add your code here
    train_data, train_labels, test_data, test_labels = split_data(features, labels, ratio=0.7)

    print('Train set shape:', train_data.shape)
    print('Train labels shape:', train_labels.shape)
    print('Test set shape:', test_data.shape)
    print('Test labels shape:', test_labels.shape)
    
    
    # Fit your model
    # add your code here
    model  = GridSearchCV(classifierType(**hp),hyperparameters, n_jobs = -1,refit = "presicion_score")

   
    clf  = model.fit(train_data,train_labels)
   
    # Make predictions for training dataset
    # add your code here
    predictions_train = clf.predict(train_data)
    # Compute train FAR/FRR
    # add your code here
    trainFAR, trainFRR = get_metrics(train_labels, predictions_train)

    # Make predictions for testing dataset
    # add your code here

    predictions_test = clf.predict(test_data)

    # Compute test FAR/FRR
    # add your code here
    testFAR, testFRR = get_metrics(test_labels,predictions_test)
    print("\tbest params are ",clf.best_params_)
    return trainFAR, trainFRR, testFAR, testFRR


trainFAR, trainFRR, testFAR, testFRR = evaluate(classifierType[0], hyperparameters, features, labels,hp)
print('Train:')
print('\tFAR:', trainFAR)
print('\tFRR:', trainFRR)

print('Test:')
print('\tFAR:', testFAR)
print('\tFRR:', testFRR)
#------------------------------------------------------------------------
print("   SVM: ")
hyperparameters=  [ 
                   {
                      'C': [10,20,15,5,9,13,25,50, 100,500, 1000], 
                      'gamma': [0.001,0.00075,0.0009,0.0005,0.0008,0.0015,0.0017,0.00173,0.0016,0.012,0.0175,0.002,0.0025,0.0015,0.003], 
                      'kernel': ['rbf']
                    }]
              
hp =    {             'C': 1, 
                      'kernel': 'linear'
                    }
trainFAR, trainFRR, testFAR, testFRR = evaluate(classifierType[2], hyperparameters, features, labels,hp)

print('Train:')
print('\tFAR:', trainFAR)
print('\tFRR:', trainFRR)

print('Test:')
print('\tFAR:', testFAR)
print('\tFRR:', testFRR)
print("FINISHED")
    
