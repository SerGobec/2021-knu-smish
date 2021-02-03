from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')

import spacy

import statsmodels.api as sm

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report

import joblib

import os.path

class SMSBase:
    # Spacy library is loading English dictionary.
    _nlp = spacy.load("en_core_web_sm")
    
    def __init__(self, filename, frac=0.8):
        self._filename = filename
        self._features = ['class', 'context']
        
        self._df_raw = pd.read_csv(self._filename, sep='\t', names=self._features)
        self.__format_context()
        
        self.__extract_features()
        
        self._group_by_feature = self._df_raw .groupby('class')
        self._counts_by_features = self._group_by_feature.count().to_dict()['context']
        
        self.__split_test_train(frac)
        
    def __format_context(self):
        self._df_raw['context'] =  self._df_raw['context'].map(lambda text : text.rstrip())
        self._df_raw['context'] =  self._df_raw['context'].map(lambda text : text.replace(',', ' ,') if ',' in text else text)
    
    def __extract_features(self):
        self._df_raw['len']= self._df_raw['context'].map(lambda text : len(text))
        self._df_raw['n_words'] = self._df_raw['context'].map(lambda text : len(text.split(' ')))

        #updating features
        self._features = self._df_raw.columns
    
    def __split_test_train(self, frac):
        self._df_train = self._df_raw.sample(frac=frac)
        self._df_test = self._df_raw.drop(self._df_train.index)
    
    def describe(self):
        print('-' * 20 + 'Extended Dataset (Head)' + '-' * 20)
        display(self._df_raw.head())
        
        print('-' * 20 + 'Extended Dataset (Describe)' + '-' * 20)
        display(self._df_raw.describe())
        
        print('-' * 20 + 'Groupby Class (Describe)' + '-' * 20)
        display(self._group_by_feature.describe())
        
    def create_lemmas(self, c):
        tokens = self._nlp(c)
        return [token.lemma_ for token in tokens]
    
    def create_tokens(self, c):
        tokens = self._nlp(c)
        return [token for token in tokens]
    
    
class Util:
        
    def report_classification(model, df_train, df_test, X_features, y_feature):
        
        classes_train = np.unique(df_train[y_feature].values).tolist()
        classes_test = np.unique(df_test[y_feature].values).tolist()
        
        assert (classes_train == classes_test)
        
        classes = classes_train # The order of class is important!
        
        X_train = df_train[X_features].values.tolist()
        X_test = df_test[X_features].values.tolist()
        
        y_train = df_train[y_feature].values.tolist()
        y_test = df_test[y_feature].values.tolist()
        
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        Util.report_cm(y_train, y_test, y_train_pred, y_test_pred, classes)
        
    def report_cm(y_train, y_test, y_train_pred, y_test_pred, classes):
        figure, axes = plt.subplots(1, 2, figsize=(10,5))

        cm_test = confusion_matrix(y_test, y_test_pred)
        df_cm_test = pd.DataFrame(cm_test, index = classes, columns = classes)
        ax = sns.heatmap(df_cm_test, annot=True, ax = axes[0], square= True)
        ax.set_title('Test CM')

        cm_train = confusion_matrix(y_train, y_train_pred)
        df_cm_train = pd.DataFrame(cm_train, index = classes, columns = classes)
        ax = sns.heatmap(df_cm_train, annot=True, ax = axes[1], square= True)
        ax.set_title('Train CM')

        print('-' * 20 + 'Testing Performance' + '-' * 20)
        print(classification_report(y_test, y_test_pred, target_names = classes))
        print('acc: ', metrics.accuracy_score(y_test, y_test_pred))

        print('-' * 20 + 'Training Performance' + '-' * 20)
        print(classification_report(y_train, y_train_pred, target_names = classes))
        print('acc: ', metrics.accuracy_score(y_train, y_train_pred))
        
    
    def plot_cdf(p, 
             ax, 
             deltax=None, 
             xlog=False, 
             xlim=[0, 1], 
             deltay=0.25, 
             ylog=False, 
             ylim=[0,1], 
             xlabel = 'x'):

        df = pd.DataFrame(p, columns=[xlabel])
        display(df.describe())
        
        ecdf = sm.distributions.ECDF(p)
        x = ecdf.x
        y = ecdf.y
        assert len(x) == len(y)
        if deltax is not None:
            x_ticks = np.arange(xlim[0], xlim[1] + deltax, deltax)
            ax.set_xticks(x_ticks)

        ax.set_xlabel(xlabel)
        ax.set_xlim(xlim[0], xlim[1])
        ax.vlines(np.mean(p), min(y), max(y), color='red', label='mean', linewidth=2)
        ax.vlines(np.median(p), min(y), max(y), color='orange', label='median', linewidth=2)
        ax.vlines(np.mean(p) + 2 * np.std(p), min(y), max(y), color='blue', label='mean + 2 * std', linewidth=2)
        ax.vlines(np.mean(p) + 3 * np.std(p), min(y), max(y), color='green', label='mean + 3 * std', linewidth=2)

        y_ticks = np.arange(ylim[0], ylim[1] + deltay, deltay)
        ax.set_ylabel('CDF')
        ax.set_yticks(y_ticks)
        ax.set_ylim(ylim[0], ylim[1])

        if xlog is True:
            ax.set_xscale('log')

        if ylog is True:
            ax.set_yscale('log')


        ax.grid(which='minor', alpha=0.5)
        ax.grid(which='major', alpha=0.9)

        ax.legend(loc=4)

        sns.set_style('whitegrid')
        sns.regplot(x=x, y=y, fit_reg=False, scatter=True, ax = ax)
    
        
    def plot_class_dist(df, by):
        
        x_features = df.columns.drop(by)
        assert 0 < len(x_features)
        
        x_features = x_features[0]
        dist = df.groupby(by)[x_features].size() / len(df)
        display(dist)        
        sns.barplot(x=dist.index, y=dist.values)
        
    def plot_boxplot(df, by, y, ax):
        ax = sns.boxplot(x=by, y=y, data=df[[by,  y]], ax = ax)
        ax.set_yscale('log')
        
    def dump_pickle(obj,filename):
        joblib.dump(obj, filename)
        
    def load_pickle(filename):
        return joblib.load(filename)
    
class SMSClassification(SMSBase):
    __pipelines = {}
    __params = {}
    __format_model_file_name = '{}_model.pkl'

    def __init__(self, filename, frac=0.8):
        super().__init__(filename, frac)
        
        self.__bow = CountVectorizer(analyzer=self.create_lemmas)
        self.__tfidf = TfidfTransformer()
        
        self.__svd = TruncatedSVD(n_components=50)

        self.__cv = StratifiedKFold(n_splits=10)
        
        self.__default_params = {
            'tfidf__use_idf': (True, False),
            'bow__analyzer': (self.create_lemmas, self.create_tokens),
        }
        
        self.__X = self._df_train['context'].values.tolist()
        self.__y = self._df_train['class'].values.tolist()
        
   
    def __create_pipeline(self, option='NB'):
                        
        if (option in self.__pipelines) is False:
                        
            if option is 'NB':
                classifier = MultinomialNB()
                pipeline = Pipeline([
                    ('bow', self.__bow),
                    ('tfidf', self.__tfidf),
                    ('classifier', classifier),
                ])

            elif option is 'SVM':
                classifier = SVC()
                pipeline = Pipeline([
                    ('bow', self.__bow),
                    ('tfidf', self.__tfidf),
                    ('svd', self.__svd),
                    ('classifier', classifier),
                ])
                
            elif option is 'RFT':
                classifier = RandomForestClassifier()
                pipeline = Pipeline([
                    ('bow', self.__bow),
                    ('tfidf', self.__tfidf),
                    ('svd', self.__svd),
                    ('classifier', classifier),
                ])
                
            else:
                classifier = MultinomialNB()

            self.__pipelines[option] = pipeline
            
            return pipeline

        else:
            return self.__pipelines[option]
            
            
    def __create_grid_search_params(self, option='NB'):
        
        if (option in self.__params) is False:
            if option is 'SVM':
                params = [
                    {
                      'classifier__C': [1, 10, 100, 1000], 
                      'classifier__kernel': ['linear']
                    },
                    {
                      'classifier__C': [1, 10, 100, 1000], 
                      'classifier__gamma': [0.001, 0.0001], 
                      'classifier__kernel': ['rbf']
                    },
                ]

                # merging two list of paramaters on the same list.
#                 params = list(map(lambda m : {**m, **self.__default_params}, params))
            else:
                params = self.__default_params

            self.__params[option] = params
        else:
            params = self.__params[option]
            
        return params

        
        
    def validate(self, option='NB'):
        
        pipeline = self.__create_pipeline(option)
        if pipeline is not None:            
            scores = cross_val_score(pipeline, 
                                     self.__X, 
                                     self.__y, 
                                     scoring='accuracy', 
                                     cv=self.__cv, 
                                     verbose=1, 
                                     n_jobs=1)

            print('scores={}\nmean={} std={}'.format(scores, scores.mean(), scores.std()))
        else:
            print ("pipeline does not exist!")

        
    def train(self, option='NB', dump=True):
        
        pipeline = self.__create_pipeline(option)
        if pipeline is not None:
            
            params = self.__create_grid_search_params(option)
            
            grid = GridSearchCV(
                pipeline, 
                params, 
                refit=True, 
                n_jobs=1, 
                scoring='accuracy', 
                cv=self.__cv)

            model = grid.fit(self.__X, self.__y)
            
            display('(Grid Search) Best Parameters:', )
            display(pd.DataFrame([model.best_params_]))

            if dump:
                model_file_name = self.__format_model_file_name.format(option)
                Util.dump_pickle(model, model_file_name)
                
            return model
                
        else:
            print('pipeline does not exist!')
            return None

    
    def test(self, X=None, model=None, model_file=None):
        
        if X is None:
            X = self.__X
        
        if model is None and model_file is None:
            print('Please, use either model or model_file')
            return []
        
        if model_file is not None and os.path.isfile(model_file):
            model = Util.load_pickle(model_file)
            print('{} file was loaded'.format(model_file))
            return model.predict(X)
        
        if model is not None:
            return model.predict(X)
        else:
            return []



sms = SMSClassification('data.txt')

sms.describe()


classifiers = ['NB', 'SVM', 'RFT']
for c in classifiers:
    display('-' * 40 + c + '-' * 40)
    sms.validate(c)

models = {}

for c in classifiers:
    display('-' * 40 + c + '-' * 40)
    models[c] = sms.train(c)

for c in classifiers:
    display('-' * 40 + c + '-' * 40)
    model_file = '{}_model.pkl'.format(c)
    r = sms.test(model_file=model_file)
    display(r)

for c in classifiers:
    display('-' * 40 + c + '-' * 40)
    model_file = '{}_model.pkl'.format(c)
    model = Util.load_pickle(model_file)
    Util.report_classification(model, 
                               sms._df_train, 
                               sms._df_test, 
                               'context', 
                               'class')
