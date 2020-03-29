# Larry To
# Created on: 3/14/2020

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle


# Loading Json files 
with open("intents.json") as file:
	data = json.load(file)


# print(data["intents"])
try: 
	with open("data.pickle", "rb") as f:
		words, labels, training, output = pickle.load(f)

except:
	words = []
	labels = []
	docs = []
	docs_x = []
	docs_y = [] 

	for intent in data["intents"]:
		for pattern in intent["patterns"]:
			wrds = nltk.word_tokenize(pattern)
			words.extend(wrds)

			docs_x.append(wrds)
			docs_y.append(intent["tag"])

			if intent["tag"] not in labels:
				labels.append(intent["tag"])

	words = [stemmer.stem(w.lower()) for w in words if w != "?"]
	words = sorted(list(set(words))) # sort and remove duplicate, then convert to list 

	labels = sorted(labels)

	# create a "one hot encoding"/bag of words 
	training = []
	output = []

	out_empty = [0 for _ in range(len(labels))]

	for x, doc in enumerate(docs_x):
		bag = []

		wrds = [stemmer.stem(w) for w in doc]

		for w in words:
			if w in wrds:
				bag.append(1)
			else:
				bag.append(0)

		output_row = out_empty[:]
		output_row[labels.index(docs_y[x])] = 1

		training.append(bag)
		output.append(output_row)

	training = numpy.array(training) # training data
	output = numpy.array(output)

	# Saving training data 
	with open("data.pickle", "wb") as f:
		pickle.dump((words, labels, training, output), f)


# Building Model 
tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8) # layer 1 with 8 neurons 
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") # softmax activation function gives probability to all output neurons(responses)
net = tflearn.regression(net)

model = tflearn.DNN(net)

# fitting model 
# try:
# 	model.load("model.tflearn")
# except:
# 	model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
# 	model.save("model.tflearn")

model.fit(training, output, n_epoch=2000, batch_size=8, show_metric=True)
model.save("model.tflearn")

# Classify new words 


def bag_of_words(s, words):
	bag = [0 for _ in range(len(words))]

	s_words = nltk.word_tokenize(s)
	s_words = [stemmer.stem(word.lower()) for word in s_words]


	for se in s_words:
		for i, w in enumerate(words):
			if w == se:
				bag[i] = 1

	return numpy.array(bag)

def chat():
	print("Start talking with the bot (Type quit to stop)!")
	while True:
		inp = input("You: ")
		if inp.lower() == "quit":
			break

		results = model.predict([bag_of_words(inp, words)])
		results_ind = numpy.argmax(results)
		tag = labels[results_ind]

		for tg in data["intents"]:
			if tg['tag'] == tag:
				responses = tg['responses']

		print(random.choice(responses))


chat() 













