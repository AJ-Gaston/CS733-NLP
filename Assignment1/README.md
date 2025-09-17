Assignment 1 for ODU CS 733 - Natural Language Processing

Author: Alexji Gaston

The first cell is pip installations because my computer doesn't have these packages/libraries. The first cell does not have to be run for the rest of this assignment.

The second cell is for importing these libraries for the rest of the cells. If you do not run the second cell, the third cell doesn't complete.

I dropped the first column of all-data.csv because I felt that the sentiments would be popular and create confusion when trying to get the probabilities.

I originally tried to get the data from all-data.csv as a pandas dataframe, but it was hard due to not understanding the type I was passing for n-gram generation in cell 4. I tried to create a tuple or list of tuple of strings. It got harder when I was trying to get a frequency distribution of the n-grams and when I was trying to get the word cloud visualization. 

I went with the csv file and converted the read file into a list of strings to make extraction easier. Then, convert the list of n-gram frequency into a dictionary since it already has a key and value of sorts. Cells 5,6, and 7 will generate the word clouds after running the first 4 cells since it depends on the n-gram generation model.

For task 2, I specifically used the pandas dataframe library to read in the csv file. Then drop the first column and do regular expression processing on it. In order to avoid any confusion, I created the training/test set as strings and iterate over each row as a line in the function n_gram_model. The calculate perplexity function was added on top because I orignally added the print functions found in cell 9 inside cell 8, but due to the multple print functions commented out I added it to the next cell for simplicity.


I added start and end sentence tokens since it was counted when I looked back at the textbook and used it when training and testing the model afterwards. 

The perplexity function uses the log function because I needed a way to handle 0% probability, so I used 1e-6 in the dictionary get function to get the probability to the 6th decimal place. Then I used the log function because that's what is used to avoid overflow/underflow with small probabilities. By computing the using math.exp, I'm using the natural number e and -log of the summation of probabilities, over the total number of words. This cancels out to become the geomtric mean of the inverse conditonal probabilities.

Originally, cell 10 and cell 11 were one cell. However, this would take over 30 minutes for the jupyter notebook to run nad sometimes it crashed so I had to split them to save time and memory. The markdown cell underneath is to see the difference between perplexity of the original model and the model with laplace smoothing.

Cell 11 is for the interpolation, which relies on the conditional probability without lapalce smoothing because I didn't see it in the instructions and I tried to read how it was computed in textbook. I assumed that the interpolated model runs with all three probabilities so I couldn't run it for just unigram, bigram, and trigram. i set it up to 3 because that's the final one. So, I ran it after running the rest of the cells because p1, p2, p3 are dependent on the conditional probabilities computed earlier.

NOTE: To be fair for this assignment, I wasn't familiar with the nltk library or the wordcloud library so I had to look at some resources for better understanding.

Resources that helped me to complete task(s):
https://www.nltk.org/api/nltk.tokenize
https://www.geeksforgeeks.org/python/generating-word-cloud-python/
https://stackoverflow.com/questions/67594924/adding-start-and-end-tokens-to-lines-of-a-tokenized-document
