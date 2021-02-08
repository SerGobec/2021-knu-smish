import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import numpy as np
import pandas as pd
import sklearn
import re
import whois
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
from urllib.parse import urlparse
import urllib.request

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import PassiveAggressiveClassifier



# location of dataset
DATA_PATH = r"C:\Users\User\Documents\Git\yaroslav tomek\data\dataset.txt"

# names to label-column and text-column
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

#dataset = dataset[((dataset[COLUMN_LABEL] == LABEL_LEGIT) | (dataset[COLUMN_LABEL] == LABEL_SMISHING))]

#перевірити дату реєстарції домену
def check_date(url):
    domain_name = urlparse(url).netloc
    #print('domain:', domain_name)
    whois_info = whois.whois(domain_name)
    #print(whois_info.creation_date)
    d2 = date.today()
    try:
        d1 = date(whois_info.creation_date.year,whois_info.creation_date.month,whois_info.creation_date.day)
    except AttributeError:
        try:
            d1 = date(whois_info.creation_date[0].year,whois_info.creation_date[0].month,whois_info.creation_date[0].day)
        except TypeError:
            return 1
    except TypeError:
        return 1
    result = abs(d2-d1).days
    #print ('{} days between {} and {}'.format(result, d1, d2))
    if result <186:
        return 1
    else:
        return 0

#print(check_date(url))

#знайти посилання в смс
def find(string):
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    return url[0][0]
#print(find(url))

#перевірити чи є form в сарс коді сайту
def check_form(url):
    user_agent = 'Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.9.0.7) Gecko/2009021910 Firefox/3.0.7'
    headers = {'User-Agent': user_agent, }
    request = urllib.request.Request(url, None, headers)  # The assembled request
    try:
        response = urllib.request.urlopen(request)
    except urllib.error.HTTPError:
        #print('123')
        return 1
    except urllib.error.URLError:
        return 0
    data = response.read()  # The data u need

    if '<form ' in str(data):
        return 1
    else:
        return 0


def messages2vectors(messages, size):
    features = np.zeros((size, 14))
    n=0 #індекс меседжу
    greetings = ['hey ', 'Hey','hi ' ,'Hi ', 'Hello', 'hello', 'Good '] #вітання в смс
    feelings = ['annoy', 'furious ', 'hate', 'upset', 'disgusted', 'shy', 'uncertain', 'frustrated', 'scare',
                'nervous', ':)', ':('] #емоції в смс
    symbols = ['+', '-', '%', '@', '^', '/', '$', '<', '>', '{', '[', ']', '|', '}', '#', '*'] #підозрілі символи
    selfanswers = ['click', 'follow', 'register ', 'find', 'decline', 'visit', 'go ', 'call ', 'reply', 'send', 'claim',
                   'regist'] #накази в смс
    smissymb = ['$', '£' , '€', 'UAH'] #знаки грошей
    smiskeys = ['award', 'congratulations', 'winner', 'alert', 'claim', 'activate', 'verify', 'attempts', 'gift', 'voucher',
                'blocked', 'suspend', 'unlock', 'won', 'prize', 'subscribe', 'activity', 'update', 'coupon', 'refund', 'free',
                'ALERT', 'Alert', 'Win', 'Activate', 'Verify', 'GIFT', 'Gift', 'PRIZE', 'Prize', 'FREE', 'Free', 'WON',
                'BLOCKED', 'UNLOCK', 'CLAIM', 'Claim', 'SUSPEND', 'UPDATE', 'Update', 'attention', 'Attention',
                'ATTENTION', 'WIN', 'locked', 'validat', 'restore', 'Restore', 'RESTORE', 'verify', 'link', 'frozen',
                'freeze', 'unfreeze', 'FROZEN', 'locked', 'LOCKED', 'accept', 'ACCEPT', 'money', 'congrats', 'Congrats',
                'activity', 'personal', 'Personal', 'PERSONAL'] #ключові слова для смішингу
    for vector in features:
        # перша та друга фіча за замовчуванням 1, а не 0, бо вони стосуються legi sms
        vector[0]=1
        vector[1]=1
        for gre in greetings:
            #якщо наявне привітання, то перша фіча 0, тобто легальна
            if gre in messages[n]:
                vector[0]=0
        for fee in feelings:
            #якщо наявна емоція, то друга фіча 0, тобто легальна
            if fee in messages[n]:
                vector[1]=0
        #якщо є лінк в смс, то третя фіча 1, тобто смішинг
        if 'http' in messages[n]:
            vector[2]=1
        #якщо є мат. символи в смс, то четверта фіча 1
        for sy in symbols:
            if sy in messages[n]:
                vector[3]=1
        #якщо довжина смс більша 140, то п'ята фіча 1
        if len(messages[n])>140:
            vector[4]=1
        #якщо є селф-ансерс, то шоста фіча 1
        for sel in selfanswers:
            if sel in messages[n]:
                vector[5]=1
        #якщо є смішинговй символ, то сьма фіча 1
        for smsy in smissymb:
            if smsy in messages[n]:
                vector[6]=1
        #якщо є ключове смішингове слово, то восьма фіча 1
        for smkey in smiskeys:
            if smkey in messages[n]:
                vector[7]=1
        #якщо є телефонний номер, то дев'ята фіча 1
        if re.findall(r'\d{7}', messages[n])!=[]:
            vector[8]=1
        #якщо є email, то десята фіча 1
        if re.findall(r'.@\w{2,6}\w{2,3}', messages[n])!=[]:
            vector[9]=1
        #якщо є посилання та домен молодше 6 місяців, то одинацията фіча 1
        if vector[2]==1:
            vector[10]=check_date(find(messages[n]))
            #якщо більше двох слешів в лінку, то дванадцята фіча 1
            if len(re.findall('/', find(messages[n])))>=4:
                vector[11]=1
            #якщо є тире в лінку, то тринадцята фіча 1
            if '-' in find(messages[n]):
                vector[12]=1
            #якщо є форма в сарс коді сайту, то чотирнадцята фіча 1
            #print(find(messages[n]))
            vector[13]=check_form(find(messages[n]))
        else:
            vector[10]=0



        print(vector, n)




        n+=1


    return features

#print(dataset[COLUMN_TEXT])
#features = messages2vectors(dataset[COLUMN_TEXT], dataset.shape[0])

def convert_labels(labels_raw):
    '''
    Transforms labels into numerical values;
    Parameters:
        labels_raw    -   array of text-labels;
    Returns:
        features    -   array of numerical labels;
    '''

    labels = labels_raw.replace('LEGI', 0)
    labels = labels.replace('SPAM', 0)
    labels = labels.replace('SMIS', 1)

    return labels

features = messages2vectors(dataset[COLUMN_TEXT], dataset.shape[0])
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
    tp, fn, fp, tn = confusion_matrix(labels, predictions).ravel()
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

    #print('metrics', FAR, FRR)
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

    #print('Train set shape:', train_data.shape)
    #print('Train labels shape:', train_labels.shape)
    #print('Test set shape:', test_data.shape)
    #print('Test labels shape:', test_labels.shape)

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
hyperparameters1 = {'n_estimators' :300,
                'criterion' : 'gini',
                'max_depth': 8,
                'min_samples_split' : 4,
                'min_samples_leaf': 1,
                'min_weight_fraction_leaf': 0.0,
                'max_features': 'log2',
                'max_leaf_nodes': None,
                'min_impurity_decrease': 0}
print("--RANDOM FOREST CLASSIFIER--")
print(evaluate(classifierType1, hyperparameters1, features, labels))

classifierType2 = sklearn.naive_bayes.BernoulliNB
hyperparameters2 = {'alpha':0,
                'binarize':None,
                'fit_prior':False,
                'class_prior':None,
                'fit_prior': False}
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
                    'verbose':3,
                    'loss':'hinge',
                    'n_jobs':None,
                    'random_state':None,
                    'warm_start':True,
                    'class_weight':None,
                    'average':True}

print("--PASSIVEAGRESSIVE CLASSIFIER--")
print(evaluate(classifierType3, hyperparameters3, features, labels))
