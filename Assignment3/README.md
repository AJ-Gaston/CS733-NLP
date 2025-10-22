Assignment 3 for ODU CS 733 - Natural Language Processing

Author: Alexji Gaston

The first cell imports the dataset as a csv file and then converts it into a list of strings. I decided to get rid of the second column, which contained the urls for each tedt talk. The reason was since we're focusing on the transcripts, it was easire for simplicity.

The second cell uses the regular expression(re) and nltk library for preprocessing. I used re to get rid of special characters and punctuations, while using nltk to tokenize the text and eliminate stop words since the library already had a ready made list to use.

The third cell is my sanity check to ensure that everything is working out the way I want. This cell is entirely optional.

The fourth cell is the word2vec class. The reason why it's a class instead of a function is because when I was trying to incorporate everything in the textbook, it seemed that there were a lot of components so I decided to make it a class instead with functions inside it. I used the numpy library in order to do matrix and vector operations. The cell also uses the collection library to count the number of words.

The fifth cell is where I create the co-occurence matrix for the PMI and PPMI.

The sixth cell is where the term frequency inver document frequency is computed. That

The seventh cell is where I made my word pairs. I decided to use words that would appear together, especially given the ted talks.

The eight cell is cosine similarity for each of the word embeddings.

The ninth cell is the heatmap analysis. I used 