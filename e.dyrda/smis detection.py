from IPython.display import display

# import libraries for reading data, exploring and plotting
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# library for train test split
from sklearn.model_selection import train_test_split
# deep learning libraries for text pre-processing
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Modeling 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional
import tensorflow.keras.metrics 
from keras.wrappers.scikit_learn import KerasClassifier

from keras import backend as K


messages = pd.read_csv('data.txt', sep ='\t',names=["label", "message"])
duplicatedRow = messages[messages.duplicated()]
messages.groupby('label').describe().T


ham_msg = messages[messages.label =='LEGI']
spam_msg = messages[messages.label=='SPAM']
smis_msg = messages[messages.label=='SMIS']

ham_msg_text = " ".join(ham_msg.message.to_numpy().tolist())
spam_msg_text = " ".join(spam_msg.message.to_numpy().tolist())
smis_msg_text = " ".join(smis_msg.message.to_numpy().tolist())




ham_msg_df = ham_msg.sample(n = len(smis_msg), random_state = 44)
smis_msg_df = smis_msg

msg_df = ham_msg_df.append(smis_msg_df).reset_index(drop=True)
msg_df['text_length'] = msg_df['message'].apply(len)
labels = msg_df.groupby('label').mean()


msg_df['msg_type']= msg_df['label'].map({'LEGI': 0,'SPAM': 0, 'SMIS': 1})
msg_label = msg_df['msg_type'].values
X_train, X_test, y_train, y_test = train_test_split(msg_df['message'], msg_label, test_size=0.2, random_state=434)

#constants

max_len = 50 
trunc_type = "post" 
padding_type = "post" 
vocab_size = 500
embeding_dim = 16
drop_value = 0.2 
n_dense = 24
e = 50
n_lstm = 20
drop_lstm =0.2

#tokenizing

tokenizer = Tokenizer(num_words = vocab_size, char_level=False)
tokenizer.fit_on_texts(X_train)
 
training_sequences = tokenizer.texts_to_sequences(X_train)
training_padded = pad_sequences (training_sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type )
testing_sequences = tokenizer.texts_to_sequences(X_test)
testing_padded = pad_sequences(testing_sequences, maxlen = max_len,
padding = padding_type, truncating = trunc_type)


#modeling

#==========================================================LSTM==========================================================

model = Sequential()
model.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
model.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
model.add(Dense(1, activation='sigmoid'))
model.summary()



model.compile(optimizer="adam",
              loss="binary_crossentropy",
              metrics=['accuracy', tf.keras.metrics.Precision()])


display('='* 40 + 'LSTM' + '=' * 40)


history = model.fit(training_padded, y_train, epochs=e, validation_data=(testing_padded, y_test), verbose=0)



best_val_score = max(history.history['val_accuracy'])
print('Best Score by val_accuracy: {}'.format(best_val_score))

#==========================================================Bidirectional LSTM==========================================================

model1 = Sequential()
model1.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
model1.add(Bidirectional(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True)))
model1.add(Dense(1, activation='sigmoid'))

model1.compile(loss = 'binary_crossentropy', optimizer = 'nadam', metrics=['accuracy'])

display('=' * 40 + 'Bidirectional LSTM' + '=' * 40)


history1 = model1.fit(training_padded, y_train, epochs=e, validation_data=(testing_padded, y_test), verbose=0)
best_val_score1 = max(history1.history['val_accuracy'])
print('Best Score by val_accuracy: {}'.format(best_val_score1))

