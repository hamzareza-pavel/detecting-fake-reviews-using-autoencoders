#author: Hamza Reza Pavel

import re
from csv import reader
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
import statistics as st
from keras.layers import Embedding
from keras import Input
from scipy.spatial import distance


def dist_cosine(elem1, elem2):
    temp1=elem1.flatten()
    temp2=elem2.flatten()
    return distance.cosine(temp1, temp2)

def identifyFakeReviews(similarityarray, cutoffScore):
    fakereviews = []
    for i in range(len(similarityarray)):
        #print(f"similarity score for {i}th review is {similarityarray[i]}")
        if similarityarray[i] < cutoffScore:
            fakereviews.append(i)
    return fakereviews


def similarityScores(originalDF, reducedDF):
    similarities = []
    for i in range(len(originalDF)):

        result = 1 - dist_cosine(originalDF[i], reducedDF[i])
        similarities.append(result)
    print(f"Mean for similarities scores {np.mean(similarities)}")

    return similarities

def estimateCutOffScore(similarities):
    mean = st.mean(similarities)
    sd = st.stdev(similarities)
    cutoffscore = mean - sd
    print(f"mean = {mean} sd = {sd}, coscore = {cutoffscore}")
    return cutoffscore

def printConfMatrix(classifications, fakereviewindices):
    truepos = len(classifications)/2  #because our dataset consists of half pos reviews and half neg reviews
    trueneg = len(classifications)/2
    fakepos = 0
    fakeneg = 0

    for i in fakereviewindices:
        if int(classifications[i]) == 1:
            fakepos+=1
        else:
            fakeneg+=1
    truepos = truepos - fakepos
    trueneg = trueneg - fakeneg

    print(f"True pos = {truepos} Fakepos = {fakepos} \n Trueneg = {trueneg} Fakeneg = {fakeneg}")


def remove_non_vocabWords(review, vocab, istrain):
    #removes the words that are not in the vocab list
    if istrain is False:
        review = re.sub(r'<[^>]+>', ' ', review)
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = re.sub(r"\s+[a-zA-Z]\s+", ' ', review)
        review = re.sub(r'\s+', ' ', review)
    words = review.split()
    words = [w for w in words if w in vocab]
    text = ' '.join(words)
    return text

def load_vocab(filename):
    file = open(filename, 'r')
    data = file.read()
    file.close()
    vocab = data.split()
    vocab = set(vocab)
    return vocab

def read_data_from_csv(filename, vocab, istrain):
    list_of_rows = []
    with open(filename, 'r', encoding='utf-8') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)
    rating = []
    reviews = []
    classes = []
    for row in list_of_rows:
        r = row[1]
        text = remove_non_vocabWords(row[2], vocab, istrain)
        c = row[3]
        rating.append(r)
        reviews.append(text)
        classes.append(c)
    return rating, reviews, classes


def main():
    #load and tokenize the training set reviews
    vocab = load_vocab("vocab.txt")
    tr_pos_rating, tr_pos_reviews, tr_pos_classes = read_data_from_csv("train_pos.csv", vocab, True)#training set
    tr_neg_rating, tr_neg_reviews, tr_neg_classes = read_data_from_csv("train_neg.csv", vocab, True)
    train_review_set = tr_pos_reviews + tr_neg_reviews
    tr_classifications = tr_pos_classes + tr_neg_classes

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_review_set)

    encoded_reviews = tokenizer.texts_to_sequences(train_review_set)

    #maximum seq length to pad the sequences to match same length
    sequence_length = max([len(s.split()) for s in train_review_set])
    X_train = pad_sequences(encoded_reviews, maxlen=sequence_length, padding='post')
    print("Finished tokenizing traning set")

    #load and tokenize the test set reviews. seperate out the classficiation for error calc
    ts_pos_rating, ts_pos_reviews, ts_pos_classes = read_data_from_csv("test_pos.csv", vocab, False)#testing set
    ts_neg_rating, ts_neg_reviews, ts_neg_classes = read_data_from_csv("test_neg.csv", vocab, False)
    test_review_set = ts_pos_reviews + ts_neg_reviews
    ts_classifications = ts_pos_classes + ts_neg_classes

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(test_review_set)

    encoded_reviews = tokenizer.texts_to_sequences(test_review_set)

    #maximum seq length to pad the sequences to match same length
    X_test = pad_sequences(encoded_reviews, maxlen=sequence_length, padding='post')

    print("Finished tokenizing testing set")

    #vocabsize = len(tokenizer.word_index) + 1
    vocabsize = len(vocab) + 1
    print(f"length of vocabsize {vocabsize} and vocab {len(vocab)}")

    #reduce data dimension instead of actual data dimention to stop shortage of memory
    data_dim = 150
    #sttest

    model = Sequential()
    model.add(Embedding(vocabsize, data_dim))
    model.compile('rmsprop', 'mse')

    input_i = Input(shape=(1323,64))
    encoded_h1 = Dense(64, activation='relu')(input_i)
    encoded_h2 = Dense(32, activation='relu')(encoded_h1)
    encoded_h3 = Dense(16, activation='relu')(encoded_h2)
    encoded_h4 = Dense(8, activation='relu')(encoded_h3)
    encoded_h5 = Dense(4, activation='relu')(encoded_h4)
    latent = Dense(2, activation='relu')(encoded_h5)
    decoder_h1 = Dense(4, activation='relu')(latent)
    decoder_h2 = Dense(8, activation='relu')(decoder_h1)
    decoder_h3 = Dense(16, activation='relu')(decoder_h2)
    decoder_h4 = Dense(32, activation='relu')(decoder_h3)
    decoder_h5 = Dense(64, activation='relu')(decoder_h4)

    output = Dense(64, activation='relu')(decoder_h5)
    autoencoder = Model(input_i,output)

    sx_text = X_test
    X_testemb = model.predict(sx_text)

    autoencoder.compile('adam','mse',metrics=['accuracy'])
    X_embedded = model.predict(X_train)
    autoencoder.fit(X_embedded,X_embedded, epochs=10, batch_size=256, validation_data=(X_testemb, X_testemb))

    #take a sub portion of the testing data set due to lack of memory

    predicted_res = autoencoder.predict(X_testemb)
    print(f"shape of predicted res {predicted_res.shape} and {predicted_res[1].shape}")
    similarities = similarityScores(X_testemb, predicted_res)
    print(f"lengh of loss array is {len(similarities)}")

    cutoffscores = estimateCutOffScore(similarities)

    #identify the outliers ie the fake reviews
    fakervs = identifyFakeReviews(similarities, cutoffscores)
    print(f"Number of fake reviews identified {len(fakervs)}")

    printConfMatrix(ts_classifications, fakervs)
   

if __name__ == "__main__":
    main()
