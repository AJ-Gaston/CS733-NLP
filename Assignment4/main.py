import csv
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import feedforward_neural_network

def preprocessing(text):
    """
    Preprocess the text in dataset (lower the words, remove stopwords, etc.)
    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    preprocess = []
    return preprocessed_text

def accuracy():
    accuracy = 0
    return accuracy

def precision():
    precision = 0
    return precision

def f1_score():
    model_f1_score = 0
    return  model_f1_score
def perplexity():
    """
    Calculate the perplexity of a language model

    Returns:
        model_perplexity (float): an indicator of the model's perplexity
    """
    model_perplexity = 0
    return model_perplexity

def main():
    #open the ham-spam file and read it as a csv file
    with open('./training_data/ham-spam.csv', encoding='latin-1') as spamFile:
        spam_csv = csv.reader(spamFile)
        #ignore the first header bc it has the columns
        next(spam_csv)
    return 0

if __name__ == '__main__':
    main()