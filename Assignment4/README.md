# Assignment 4 for ODU CS 733 - Natural Language Processing

## Author: Alexji Gaston

Cell 1 is where I import my python libraries for the rest of the jupyter notebook. 

Cell 2 is where I read in my datasets. I used the pandas library to read in the ham-spam csv file and the os library to read in all the files inside the wikitext-2 dataset folder.

Cell 3 is a sanity check to see what types these are for later work. This is not necessary to run. That goes the same for cell 4, which checks each entry in the wiki-text list, and cell 5, which checks the pandas dataframe.

Cell 6 and cell 7 deal with preprocessing the two datasets. Cell 6 is the same as assignment 3 cell 2. I basically reused the code and created a list of list of strings for the wiki-text dataset. Cell 7 is a function that preprocesses a pandas dataframe. For both of these cells, I used the re library to use regular expression to get rid of punctuation marks or anything else. I used the inflect library like in Assingment 3 since there are numbers in the wiki-text dataset and I wasn't sure if the word2vec could handle those for the neural network.

Cell 8 is to check the pandas dataframe after preprocessing the text.



References for this assignment:
https://radimrehurek.com/gensim/models/word2vec.html
https://www.geeksforgeeks.org/nlp/word2vec-with-gensim/
