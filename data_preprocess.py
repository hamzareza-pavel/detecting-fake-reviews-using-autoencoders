#author: Hamza Reza Pavel
# this file creates a csv by cleaning, preprocessing and merging the individual review files

import os
import re
import glob
import csv
import io
from nltk.corpus import stopwords
from collections import Counter


def readRawTextForCSV(baseid, textDirectory, classification, istrainingdata, vocab):
    toCSV = []
    for filename in glob.glob(os.path.join(textDirectory, '*.txt')):
        with io.open(os.path.join(textDirectory, filename), 'r', encoding='utf8') as f:
            print(f"Processing file: {filename}")
            basefilename = os.path.basename(filename)
            token = basefilename.split(".")
            token = token[0].split("_")
            id = token[0]
            rating = token[1]
            data = f.read().replace('\n', '')
            #cleaning the text data. removing punctuations, tags, numbers, multiple spaces, and digits. do this phase only if its training data
            if istrainingdata:
                data = data.lower()
                data = cleanTextData(data, vocab)
            entry = [baseid + int(id), rating, data, classification]
            toCSV.append(entry)
    return toCSV

def cleanTextData(rawTextData, vocab):
    # Removing html tags, punctuations, numbers, single character and multiple spaces from the reviews.
    cleanedText = re.sub(r'<[^>]+>', ' ', rawTextData)
    cleanedText = re.sub('[^a-zA-Z]', ' ', cleanedText)
    cleanedText = re.sub(r"\s+[a-zA-Z]\s+", ' ', cleanedText)
    cleanedText = re.sub(r'\s+', ' ', cleanedText)

    # remove the stop words from the text.
    stop = stopwords.words('english')
    wordlist = [i for i in cleanedText.split() if i not in stop]

    #filter out word less than 2 or equal to 2 chars
    wordlist = [word for word in wordlist if len(word) > 2]
    vocab.update(wordlist)
    cleanedText = ' '.join(wordlist)
    return cleanedText

def writeProcessedCSV(listtowrite, filename):
    file = io.open(filename, 'a+', newline ='', encoding='utf8')
    # writing the data into the file
    with file:
        write = csv.writer(file)
        write.writerows(listtowrite)
        print("Finished writing processed data to CSV")


def createCSVs():

    vocablist = Counter()
    print("Starting preprocessing training data...")
    subdir = "train/pos/"
    posfiles = os.path.realpath(
    os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), subdir))
    pos_sentiment_rows = readRawTextForCSV(0, posfiles, "1", True, vocablist)
    writeProcessedCSV(pos_sentiment_rows, "train_pos.csv")

    subdir = "train/neg/"
    negfiles = os.path.realpath(
    os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), subdir))
    neg_sentiment_rows = readRawTextForCSV(len(pos_sentiment_rows), negfiles, "0", True, vocablist)
    writeProcessedCSV(neg_sentiment_rows, "train_neg.csv")

    print("Starting preprocessing testing data...")
    subdir = "test/pos/"
    posfiles = os.path.realpath(
    os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), subdir))
    pos_sentiment_rows = readRawTextForCSV(0, posfiles, "1", False, vocablist)
    writeProcessedCSV(pos_sentiment_rows, "test_pos.csv")

    subdir = "test/neg/"
    negfiles = os.path.realpath(
    os.path.join(os.path.join(os.getcwd(), os.path.dirname(__file__)), subdir))
    neg_sentiment_rows = readRawTextForCSV(len(pos_sentiment_rows), negfiles, "0", False, vocablist)
    writeProcessedCSV(neg_sentiment_rows, "test_neg.csv")



    print(f"vocablist size before filtering {len(vocablist)}")
    #number of time a word must appear to be include in vocab
    minoccurance = 5
    words = [k for k,v in vocablist.items() if v>= minoccurance]
    print(f"vocablist size after filtering {len(words)}")

    #write the vocabs to file
    data = '\n'.join(words)
    file=open("vocab.txt",'w')
    file.write(data)
    file.close

createCSVs()

