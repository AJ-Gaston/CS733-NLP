# Assignment 4 for ODU CS 733 - Natural Language Processing

## Author: Alexji Gaston

Cell 1 is where I import my python libraries for the rest of the jupyter notebook. 

Cell 2 is where I read in my datasets. I used the pandas library to read in the ham-spam csv file and the os library to read in all the files inside the wikitext-2 dataset folder.

Cell 3 is a sanity check to see what types these are for later work. This is not necessary to run. That goes the same for cell 4, which checks each entry in the wiki-text list.

Cell 5 deals with preprocessing the wiki-text dataset . Cell 5 is the same as assignment 3 cell 2. I basically reused the code and created a list of list of strings for the wiki-text dataset. For this cells, I used the re library to use regular expression to get rid of punctuation marks and the inflect library to get rid of any numbers by turning them into actual words (6 into 'six', 1000 into 'one thousand', etc.) because I wasn't sure if the word2vec could handle that when converting the text into vectors. I used the nltk library's corpus to import stop words, which gets rid of stopwords from the corpus.

Cell 6 is another sanity check to see if the function worked properly.

For cell 7, this is a function that preprocesses a pandas dataframe. I used panda dataframe's apply function and I looked at the second column of the dataframe, the Text column, for preprocessing. Since there aren't any numbers here, I just used the re library, for regular expression, and the nltk library, to get rid of stop words. 

Cell 8 is to check the pandas dataframe after preprocessing the text. Basically a sanity check.

Cell 9 is where I apply word2vec to the dataframe. I downloaded and imported the gensim library, which is where the word2vec is located.

Cell 10 is 




References for this assignment:
https://radimrehurek.com/gensim/models/word2vec.html
https://www.geeksforgeeks.org/nlp/word2vec-with-gensim/
