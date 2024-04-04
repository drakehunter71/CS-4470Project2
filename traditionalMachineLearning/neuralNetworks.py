# import pickle
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split

# from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Embedding
# from tensorflow.keras.layers import Dense, Input, GlobalMaxPool1D
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.layers import Dense,Input,GlobalMaxPooling1D
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding
# from tensorflow.keras.models import Model
# from tensorflow.keras.losses import SparseCategoricalCrossentropy

# DID NOT USE
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import (
    LSTM,
    Embedding,
    Dense,
    Input,
    GlobalMaxPooling1D,
    Conv1D,
    MaxPooling1D,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("Data/combined.csv")
df["data"] = df["data"].str.replace(r"[.\s]+$", "", regex=True)
df = df[df["type"] == "tok"]
test_df = df[df["traintest"] == "test"]
train_df = df[df["traintest"] == "train"]


# Tokenization and Padding
MAX_VOCAB_SIZE = 2000
tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)
tokenizer.fit_on_texts(df["data"])
sequences = tokenizer.texts_to_sequences(train_df["data"])
data = pad_sequences(sequences, maxlen=100)  # Adjust maxlen as needed

# Splitting the dataset
y_train = train_df["speculative"].values
y_test = test_df["speculative"].values

# Number of unique words
V = len(tokenizer.word_index)
# Embedding dimension
D = 20


# CNN Model for Binary Classification
def build_cnn_model(V, D):
    i = Input(shape=(100,))  # Adjust input shape based on your padded sequences' length
    x = Embedding(V, D)(i)
    x = Conv1D(32, 3, activation="relu", padding="same")(x)  # Use padding='same'
    x = MaxPooling1D(2)(x)  # Consider reducing pool size if sequences are short
    x = Conv1D(64, 3, activation="relu", padding="same")(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(i, x)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# model_cnn = build_cnn_model(V, D)
# r_cnn = model_cnn.fit(
#     train_df["data"], y_train, epochs=10, validation_data=(test_df["data"], y_test)
# )


# RNN Model (LSTM) for Binary Classification
def build_rnn_model(V, D):
    i = Input(shape=(100,))
    x = Embedding(V + 1, D)(i)
    x = LSTM(15, return_sequences=True)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(i, x)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


model_rnn = build_rnn_model(V, D)
r_rnn = model_rnn.fit(
    test_df["data"], y_train, epochs=10, validation_data=(test_df["data"], y_test)
)

# Save the models
model_cnn.save("Results/cnn_classifier_modek_tok.h5")
model_rnn.save("Results/rnn_classifier_model_tok.h5")

# Reload models (if needed)
# cnn_loaded = load_model('models/speculative_cnn_model.h5')
# rnn_loaded = load_model('models/speculative_rnn_model.h5')

# Evaluation
y_pred_cnn = (model_cnn.predict(X_test) > 0.5).astype("int32")
print("CNN Model Evaluation")
print(classification_report(y_test, y_pred_cnn))

y_pred_rnn = (model_rnn.predict(X_test) > 0.5).astype("int32")
print("RNN Model Evaluation")
print(classification_report(y_test, y_pred_rnn))


# df = pd.read_csv("Data\combined.csv")
# df["data"] = df["data"].str.replace(r"[.\s]+$", "", regex=True)
# test_df = df[df["traintest"] == "test"]
# train_df = df[df["traintest"] == "train"]

# # Number of classes
# K = df['targets'].max() + 1
# K
# from sklearn.model_selection import train_test_split

# random_seed = 42

# df_train, df_test = train_test_split(df, test_size=0.3, random_state=random_seed)
# # Convert sentences to sequences - keeps 2000 most frequent and discard the rest.
# MAX_VOCAB_SIZE = 2000
# tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE)

# # Builds the vocabulary based on the most common words. It assigns a unique integer to each word based on its frequency
# tokenizer.fit_on_texts(df_train['text'])

# # Each word in the sentences is replaced by its corresponding integer based on the vocabulary created by the tokenizer.
# sequences_train = tokenizer.texts_to_sequences(df_train['text'])
# sequences_test = tokenizer.texts_to_sequences(df_test['text'])
# # get total unique words mapped by tokenizer (True size of our unique words)
# word2idx = tokenizer.word_index
# V = len(word2idx)
# print(f'Found unique words: {V}')
# # pad sequences so we get a N x T matrix - all the text must have same dimension size
# data_train = pad_sequences(sequences_train)
# print(f'Shape of data train tensor: {data_train.shape}')
# data_train
# # Get sequences length
# T = data_train.shape[1]
# T
# data_test = pad_sequences(sequences_test, maxlen=T)
# print(f'Shape of data test tensor: {data_test.shape}')
# # Creating the CNN model

# # Choosing embedding dimensionality
# D = 30

# i = Input(shape=(T,))
# x = Embedding(V+ 1 ,D)(i)
# x = Conv1D(32,2, strides=1 ,activation='relu')(x)
# x = MaxPooling1D(3)(x)
# x = Conv1D(64,3, activation='relu')(x)
# x = GlobalMaxPooling1D()(x)
# x = Dense(K)(x) # where K is the total of news's labels

# model = Model(i,x)

# # Compile and Fit the model
# model.compile(
#     loss=SparseCategoricalCrossentropy(from_logits=True),
#     optimizer='adam',
#     metrics=['accuracy']
# )

# print('Training the model...')
# r = model.fit(
#     data_train,
#     df_train['targets'],
#     epochs = 20,
#     validation_data=(data_test, df_test['targets'])
# )

# # Checking loss per itartion
# plt.plot(r.history['loss'], label='train loss')
# plt.plot(r.history['val_loss'], label='val loss')
# plt.legend()

# # Plot accuracy per iteration
# plt.plot(r.history['accuracy'], label='train acc')
# plt.plot(r.history['val_accuracy'], label='val acc')
# plt.legend()

# from sklearn.metrics import classification_report


# predictions = model.predict(data_test)
# y_pred = np.argmax(predictions, axis=1)  # Convert logits to class labels

# # Get true labels
# y_true = df_test['targets']

# # Generate classification report
# report = classification_report(y_true, y_pred)
# print(report)

# classes = df['targets'].unique()

# from sklearn.metrics import confusion_matrix
# import seaborn as sns

# label_mapping = {
#     0: 'Colon_Cancer',
#     1: 'Lung_Cancer',
#     2: 'Thyroid_Cancer'

# }
# marks = np.arange(len(label_mapping))

# cm = confusion_matrix(y_true, y_pred)
# df_cm = pd.DataFrame(cm, index=[label_mapping[i] for i in label_mapping] , columns = [label_mapping[i] for i in label_mapping])

# plt.figure(figsize=(8, 6))
# plt.xlabel('Predicted')
# plt.ylabel('True')
# sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', cmap='Blues')

# plt.show()

# # Creating the model for RNN (LSTM)

# i = Input(shape=(T,)) # T = Sequence Length
# x = Embedding(V+1,D)(i) # symbolic representation of the input data that will be passed through the layers of the neural network
# x = LSTM(32, return_sequences=True)(x)
# x = GlobalMaxPool1D()(x)
# x = Dense(K)(x)

# model = Model(i,x)

# # Compile and fit
# model.compile(
#   loss=SparseCategoricalCrossentropy(from_logits=True),
#   optimizer='adam',
#   metrics=['accuracy']
# )


# print('Training model...')

# r_rnn = model.fit(
#   data_train,
#   df_train['targets'],
#   epochs=20,
#   validation_data=(data_test, df_test['targets'])
# )

# predictions = model.predict(data_test)
# y_pred = np.argmax(predictions, axis=1)  # Convert logits to class labels

# # Get true labels
# y_true = df_test['targets']

# # Generate classification report
# report = classification_report(y_true, y_pred)
# print(report)

# # Checking loss per itartion
# plt.plot(r_rnn.history['loss'], label='train loss')
# plt.plot(r_rnn.history['val_loss'], label='val loss')
# plt.legend();

# # Plot accuracy per iteration
# plt.plot(r_rnn.history['accuracy'], label='train acc')
# plt.plot(r_rnn.history['val_accuracy'], label='val acc')
# plt.legend();

# label_mapping = {
#     0: 'Colon_Cancer',
#     1: 'Lung_Cancer',
#     2: 'Thyroid_Cancer'

# }
# marks = np.arange(len(label_mapping))

# cm = confusion_matrix(y_true, y_pred)
# df_cm = pd.DataFrame(cm, index=[label_mapping[i] for i in label_mapping] , columns = [label_mapping[i] for i in label_mapping])

# plt.figure(figsize=(8, 6))
# plt.xlabel('Predicted')
# plt.ylabel('True')
# sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='d', cmap='Blues')

# plt.show()

# ###
# model_pkl_file = "rnn_classifier_model.pkl"

# with open(model_pkl_file, "wb") as file:
#     pickle.dump(model, file)
