Assignment 3 for ODU CS 733 - Natural Language Processing

Author: Alexji Gaston

The first cell imports the dataset as a csv file and then converts it into a list of strings. I decided to get rid of the second column, which contained the urls for each tedt talk. The reason was since we're focusing on the transcripts, it was easire for simplicity.

The second cell uses the regular expression(re) and nltk library for preprocessing. I used re to get rid of special characters and punctuations, while using nltk to tokenize the text and eliminate stop words since the library already had a ready made list to use.

The third cell is where I create the singular corpus of the transcripts.