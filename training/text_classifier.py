import os
import json
import time
import traceback
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
import numpy as np
import pandas as pd

import tensorflow_hub as hub

""" import gensim modules"""
import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from spacy.lang.en import English
spacy_en = English(disable=['parser', 'ner'])
from spacy.lang.en.stop_words import STOP_WORDS as stopwords_en
stopwords_en = list(stopwords_en)
mapping_dict = {"en":[stopwords_en, spacy_en]} 

""" import scripts"""
from helper.cleaning_pipeline import Preprocessing
cleaning_pipeline_obj = Preprocessing()

import config as config


class ClassifyText():

	def __init__(self):
		self.oov_list=[]


	def writeToCsv(self, test_df, test_labels):
		try:
			submission = defaultdict(list)
			submission['id'].extend(test_df['id'])
			submission['label'].extend(test_labels)
			submission = pd.DataFrame(submission)
			submission.to_csv( 'submission_lstm.csv', index=False)
			print("\n saved results in csv successfully === ")
		
		except Exception as e:
			print("\n error ", e, "\n traceback === ",traceback.format_exc())


	def evaluate_model(self, test_df, model, val_x, val_Y):
		import pdb;pdb.set_trace()
		test_loss, test_acc = model.evaluate(val_x, val_Y, verbose=2)
		print('Test Loss:', test_loss)
		print('Test Accuracy:', test_acc)

		# y_pred = np.argmax(model.predict(val_x), axis=-1)
		
		# print("\n f1_score : ", f1_score(val_Y, y_pred))
		# metric = tfa.metrics.F1Score(num_classes=2, threshold=0.5)
		# metric.update_state(val_Y, y_pred)
		# result = metric.result()
		# print("\nf1 score : ", result.numpy())
		test_x = self.create_feature(test_df)
		pred = model.predict(test_x)
		y_pred = [list(pred_lst) for pred_lst in pred]
		# test_labels = np.argmax(model.predict(test_x), axis=-1)
		# self.writeToCsv(test_df, test_labels)


	def train_model(self, train_X, val_X, train_Y, val_Y):
		try:
			st = time.time()
			""" training """
			# LSTM model
			model = Sequential()
			# model.add(LSTM(300))
			model.add(Bidirectional(LSTM(300)))
			model.add(Dropout(0.3))
			model.add(Dense(2, activation='sigmoid'))
			model.compile(loss='binary_crossentropy',
			  optimizer='adam',
			  metrics=['accuracy'])

			print('Train...')
			model.fit(train_X, train_Y, batch_size=128, validation_data=(val_X, val_Y), validation_steps=30, epochs=50)
			# model.fit(train_X, train_Y, epochs=100)
			print("\n model summary : \n", model.summary())

			
		except Exception as e:
			print("\n Error in train_model : ",e)
			print("\n Error details : ", traceback.format_exc())
		return model


	def load_word2vec(self):
		st = time.time()
		self.model = KeyedVectors.load_word2vec_format(config.word2vec_model_path, limit=500000, binary=True)
		print("\n time to load word2vec model --- ", time.time() - st)
		print("\n hello vector --- \n", self.model['hello'])


	def sent_vectorizer(self, sent):
		# sent_vec = np.zeros((1,300))
		sent_vec=[]
		numw = 0
		try:
			for w in sent.split():
				try:
					sent_vec.append(self.model[w])
				except:
					self.oov_list.append(w)
					pass
			# sentence_vector = np.asarray(sent_vec) / numw
			sentence_vector = np.asarray(sent_vec)
		except Exception as e:
			print(sent_vec, numw)
			print("\n Error in sent_vectorizer : ",e)
			print("\n Error details : ", traceback.format_exc())
		return sentence_vector


	def create_w2v_vectors(self, sentences):
		self.load_word2vec()
		st = time.time()
		X = [self.sent_vectorizer(sent) for sent in sentences ]
		# X = np.array(X)
		# print("\n shape --- ", X.shape)
		print("\n time for w2v vectorization : ", time.time() - st)
		del self.model
		return X


	def get_max_length(self, reviews):
		# max_length = 0
		# for row in df['tweet']:
		# 	if len(row.split(" ")) > max_length:
		# 		max_length = len(row.split(" "))

		max_length = max([len(review.split()) for review in reviews])
		print("\n max_length:", max_length)
		return max_length


	def get_word2vec_enc(self, reviews):
		"""
		get word2vec value for each word in sentence.
		concatenate word in numpy array, so we can use it as RNN input
		"""
		embed = hub.load("https://tfhub.dev/google/Wiki-words-250/2")
		encoded_reviews = []
		for review in reviews:
			tokens = review.split(" ")
			word2vec_embedding = embed(tokens)
			encoded_reviews.append(word2vec_embedding)
		print("\n encoded_reviews shape : ", len(encoded_reviews))
		return encoded_reviews
		
			
	def get_padded_encoded_reviews(self, encoded_reviews, max_length):
		"""
		for short sentences, we prepend zero padding so all input to RNN has same length
		"""
		padded_reviews_encoding = []
		for enc_review in encoded_reviews:
			zero_padding_cnt = max_length - enc_review.shape[0]
			pad = np.zeros((1, 300))
			for i in range(zero_padding_cnt):
				enc_review = np.concatenate((pad, enc_review), axis=0)
			padded_reviews_encoding.append(enc_review)
		return padded_reviews_encoding


	def sentiment_encode(self, sentiment):
		"""
		return one hot encoding for Y value
		"""
		if sentiment == 1:return [1,0]
		else:return [0,1]


	def create_feature(self, df):
		# encode words into word2vec
		reviews = df['tweet'].tolist()
		reviews = cleaning_pipeline_obj.cleaning_pipeline(reviews)
		max_length = self.get_max_length(reviews)

		encoded_sents = self.create_w2v_vectors(reviews)
		# encoded_sents = self.get_word2vec_enc(reviews)
		padded_encoded_sents = self.get_padded_encoded_reviews(encoded_sents, max_length)
		X = np.array(padded_encoded_sents)
		print("\n shape of X : ", X.shape)
		return X


	def preprocess(self, df):
		"""
		encode text value to numeric value
		"""
		X = self.create_feature(df)
		# encoded sentiment
		labels = df['label'].tolist()
		encoded_labels = [self.sentiment_encode(label) for label in labels]
		Y = np.array(encoded_labels)
		return X, Y


	def read_data(self, data_file_path):
		train_file = os.path.join(data_file_path, "train.csv")
		test_file = os.path.join(data_file_path, "test.csv")
		train_df = pd.read_csv(train_file)
		test_df = pd.read_csv(test_file)
		return train_df, test_df 


	def main(self, data_file_path):
		try:
			print("\n inside main ::: ")
			# cleaned_sentences = cleaning_pipeline_obj.cleaning_pipeline(text_data, lang)
			# if embedding_model == "word2vec": text_vector = self.create_w2v_vectors(cleaned_sentences)
			# elif embedding_model == "bert": text_vector = self.create_bert_vectors(cleaned_sentences)
			# model = self.train_model(cleaned_sentences, text_vector, labels, number_of_clusters, lang, bot_id)
			train_df, test_df = self.read_data(data_file_path)
			X, Y = self.preprocess(train_df)
			train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=42)
			print("\n train data size: ", train_x.shape)
			print("\n validation data size: ", val_x.shape)
			model = self.train_model(train_x, val_x, train_y, val_y)
			self.evaluate_model(test_df, model, val_x, val_y)

		except Exception as e:
			print("\n Error in main : ",e)
			print("\n Error details : ", traceback.format_exc())


if __name__ == "__main__":
	obj = ClassifyText()
	data_file_path="/home/swapnil/Projects/github/Text_Classification_using_LSTM_Word2Vec/datasets/AV_sentiment_analysis"
	obj.main(data_file_path)

	"""
reference : https://blog.eduonix.com/artificial-intelligence/clustering-similar-sentences-together-using-machine-learning/
	"""