# CS 733 Semester Project

## Author: Alexji Gaston

## Project
GITHUB LINK FOR PROJECT REPOSITORY: https://github.com/AJ-Gaston/CS733-NLP/tree/main/Semester_Project

### 1. Web-scraping
For this portion I used Apify's Google Maps Review Scraper and created a csv file instead of json due to the way that Apify's Scraper works. Apify allows exporting these reviews as a .json file or a .csv file. I chose a .csv file or easier manipulation.
 
I cleaned the csv file using OpenRefine, a software previously used in CS 625 to clean datasets of information such as unneeded columns, re-ordering columns to format I like, or filling in null/nan data.

### 2. Model Implementation
For this model, I decided to use DistilBert with LightGBM instead of GloVE pre-trained, which would have been sent into bert-base-multilingual-uncased-sentiment.

The model is a class called RestaurantSentimentAnalysisModel. The initialization comes with text encoding components, which use DistilBert's Model and Tokenizer. The text encoder is frozen since DistilBERT is used for the embeddings, not the fine-tuning of the model. The tokenizer is used to get the embedding of each review, which is later looked at for textual metadata and then sent to the train_model function. The other components are used in later functions.

_get_text_embedding is a function that grabs the text embeddings of the reviews. It grabs 32 text embeddings because 7,000 reviews is still too much for DistilBERT to handle all at once, so batches are created. For this model, the max length is 128 because it was a good average for the reviews written. All of the embeddings are returned as using numpy's vstack, which converts each of the embeddings into a 2D array.

_extract_temporal_features is handles extracting temporal features of the dataset suhc as the year, month, and day of when the review was published. The year is extracted by dividing the review's date with the current year. If any of the values are missing, then the replace the missing value with 0.5 since it's in the middle. 
One issue to consider with temporal features was the months. Since the months are cyclical, that meant 2pi had to be used. So the months were divided linearly like the years, but multiplied by 2*pi. Then the sine and cosine of the months had to be calculated as well to preserve the circular distance of the months. 
Day of week is another feature that is divided by using 6, with any missing values filled with 3(Thursday). However, weekends are also important, so a is_weekend flag was created that looks at the last 2 days in the day_of_week.

_create_fetaure_names is a function that creates descriptive names for the features. This is good for later interpretability, mainly the looking at the most important features for the model.

Extract_features is a function that extracts all the features of the dataframe. The text_to_embed list is where it looks at the dataframe's text section, or the review, and checks if the review was translated or not. If the review's originalLanguage column doesn't have 'en' (meaning in English), the it grabs the translated text in the translatedLanguage column. Otherwise, the review is in English, so it appends the text. If there's not review, then '[NO TEXT]' is appended to signal a review doesn't have text to go with it. Then it grabs the text embeddings of the text.
Along with the text_embed, there's the text_source_flag which ranges from 0-2 (0 meaning it was review translated to English, 1 meaning it was the review, 2 meaning there was no review text.). This list is reshaped into a numpy array for later.
The numerical features are captured as well. These are the review's star rating, the restaurant's total score, and the discrepancy between the reviewer's rating of the restaurant and the restaurant's overall rating. These are important because they are the main things the model is trying to capture. Discrepancy is captured by subtracting the review's star from the total score. 
Temporal features are capturing with the _extract_temporal_features function.
The text metadata is captured as well. This part of the function looks at the review's length and the word count of the review. 
Then, the function looks at each restaurant's specific features. It creates a numpy array of shape(length of the dataframe, 0) which then looks at the review's published date and the emotional intensity of the review. For the published date, it looks at how long ago the review was published by checking the days and reshapes it. For the emotional intensity, it checks to see if the reviews have an exlcamation point('!') or a question mark('?'). The reason why is because a lot of times, people will add these to show their emotion when they write the review. Another is because I wasn't sure how to include the emojis in this model.
The restaurant features are transformed into a 2-D array using numpy's column_stack, which is then concatenated with every other feature into a NDArray called all_features. All_features is created with numpy's hstack, which stacks the features sequentially.
The features are given names with the _create_feature_names function and the function returns all_features.

Train_model is where the model is trained using the dataframe read in the main class. It checks if the sentiment column in a dataste doesn't exits, which it then creates. It extracts features from the dataset using the extract_features function and makes it the X (independt value), while the y (dependent value) is the sentiment column's value. 
After that the dataset is split into a training dataset and a validation set, using the predefined test_size and random seed arguments.
Then, the LightGBM's parameters are set for sentiment analysis. Objective, num_class, and metric are the important parameters to make sure the model's task is set. After that is the parameters for hypertuning the trianing model. Then, the training and validation set are transformed with LightGBM's Dataset, which processes the datasets for the LightGBM.
Then, the function evaluates the prediction and accuracy of the training and validation sets. The prediction is calculated using numpy's argmax while the accuracy is predicted using numpy's mean. It prints out the accuracy and returns the model.

Predict is the function that predicts the sentiment of new reviews. The arguments are the dataframe and return_probs, which is a boolean. If the model wasn't trained, then function raises a ValueError. Otherwise, it extracts features from the dataframe using extract_features. This function is where the cache initialized is used. The reason why is because sometimes reviews can repeat themselves, so cache can be used to avoid bottlenecking with DistilBert. If the cache is enabled, then cache_keys are created that stores the cache key of the dataframe's text and ratings.
However, cache is not used because of the assumption that a small dataset has more unique reviews compared to a dataset with over 100,000 reviews.
The return_probs is a boolean that checks to see if probabilities should be returned from the function. If it's true, then return the predictions. If it's false, then the function returns the argmax of the predictions.

Get_feature_importance is a function that returns the top n features. It first checks to see if the model was trained yet, and if it wasn't it raises a ValueError. If the model was trained, then lgb_model's feature_importance is used, to get the importance. Finally, a list is created of the features and their importance, which the model returns the top 10 out of the sorted list.

The model uses DistilBert and LightGBM for a few reasons:
- The dataset is small enough (under 7,700 rows). This is too small for BERT, which needs at least 100,000 reviews for fine-tuning
- The dataset also has heterogenous input, meaning just transformers alone doesn't work well with this. 
- Data size favors feature extraction instead of fine-tuning
- Reviews are in more than one language, which DistilBert can handle well.

DistilBert can capture deep semantic representation of the reviews, detect sarcasm/negation, and it can embed the reviews which is then sent to the LightGBM model.
LightGBM is needed because while it can't understand different languages, it can handle reviews with missing data such as text and dates. For the text, it can look at the review's star rating and the restaurant's total score. For the date, it can simply use other features. Another strength of LightGBM is that it can train fast.

After the sentiment analysis training is complete, it's sent back to the semester_project.py main function and the top 10 most important features printed onto the terminal/screen for analysis.

Then, each restaurant is analyzed. A review window is create with number, indicating the number of reviews to look at. The the trained model, dataframe focused on a single restaurant, and the review windo are sent to the analyze_restaurant_multiple_reviews function.

Analyze_restaurant_multiple_reviews creates a list that holds all the analysis results. It captures the restaurant's total score and goes through each number in the reviews_window. It skips if the restaurant doesn't have enough reviews. If it does have enough reviews, then the top reviews are looked at by publishedDate. The model predicts the review and returns the probabilities. Sentiment score is calculated by subtracting the negative score from the positive score, and the mean of the sentiment score is calculated with numpy's mean. The sentiment gap is calculated by subtracting restaurant's total score from the mean sentiment score.
Dynamic significant threshold is applied because with such a small sample size, random variation is much larger. This means that there needs to be more evidence for "significant change" to be applied. So if the significant threshold is above a certain floating point, that means there was a large enough gap to claim a significant change. The threhsold decreases by 0.05 as each review window increases. Significant change is calculated checking if the absolute value of the sentiment gap is larger than the threshold.
Then trends are calculated. If the sentiment gap is greater than 0.1, it means there was improvement for the restaurant's reviews. But if the sentimen gap is less than -0.1, it meant that there was a decline in the restaurant's reviews.
The result fo the review window is saved in a dictionary that contains all of these trends, which is then appened to the results list created at the beginning of the function.
If the results has review window analysis, then it returns a dictionary with the restaurant's title, address, total score, total reviews, window analysis, and other features. If there was no window analysis, then it returns None.

After the function, various things should be printed on the screen. It should print which restaurants have at least 1,3,5,7,10 reviews. It then prints out the window size for each n reviews followed by the restaurants analayzed(should be  63 for atleast windows 1 and 3), the average sentiment gap, significant discrepancies between the review's ratng and the restaurant's score, and prints the number of restaurants improving/declining/stable.

Then, the main method looks to see which restaurants were consistently improving and which were consistently declining. It looks at the direction key in th eresults dictionary and the sentiment gap key. Finally, it displays the top 5 consistently improving. To show it's improving, the review windows are printed along with the sentiment gap and direction of the gap.

Finally, the main saves these results with the save_detailed_results function. Save_detailed_results takes the all_results list from the main. It creates a list called rows, which contains each restaurant and review window analysis. It extracts the restaurant info and the review windows analyzed from the all_results list. The specific row copies the restaurant info and updates with the review window information. It appends the updated row to list of rows and creates a pandas Dataframe from the list, called results_df. It creates a csv file called multi_window_analysis.
Then it creates a list called summary rows. Summary_rows captures the largest review window information for each restaurant and creates a pandas Dataframe called summary_df. Then, it creates a csv file called restaurant_summary.
Both of these fields are saved to the project repository for later analysis.

## GUI
For this part, I separated created a separate file. This file creates Guided User Interface, or GUI, with tkinter library. 
In this I created three functions in the initialization: setup_style, setup_GUI, and load_data. The setup_style is where I create the fonts and colors for the GUI window. 
The setup_GUI is where I create the various frames,labels and buttons for the GUI. I create multiple frames, each of which show their own area. This is to make it easier for the user to see the restaurants, review window, total score, sentiment, and direction of sentiment.
The load data is where the class loads the csv file. While it does import from semester_project and restaurant_sentiment_analysis, it checks to see if it's available as its own functionality. 

The window is created, but the user is unable to select anything until the model has successfully trained. Once it has, the user can select any restaurant and any review window that restaurant has. Then, they click the Analyze Button to the right of the screen. It looks white until the mouse hovers over the button, which causes the text 'Analyze' to appear.
The user can look at any review window because of the clear_results function that removes the text on the window.
The window display also uses unicode/text images. This is to make it more intepretable for the user.

Throughout this file, I'm using threading. The reason why is because my GUI intially froze my computer and mouse with the training. I tried to find a few ways to resolve this and threading was the best option at the time.

### References 
#### Web-Scraping
https://youtu.be/Qrt6Jm0uOsE?si=Qcj7uk22wE5f6AxH
https://blog.apify.com/how-to-scrape-google-reviews/

#### DistilBert
https://huggingface.co/docs/transformers/en/model_doc/distilbert#transformers.DistilBertTokenizer
https://huggingface.co/docs/transformers/en/model_doc/distilbert#transformers.DistilBertModel
https://medium.com/huggingface/distilbert-8cf3380435b5
https://www.geeksforgeeks.org/nlp/distilbert-in-natural-language-processing/

#### LightGBM
https://medium.com/@sebastiencallebaut/classifying-tweets-with-lightgbm-and-the-universal-sentence-encoder-2a0208de0424
https://www.geeksforgeeks.org/machine-learning/multiclass-classification-using-lightgbm/
https://lightgbm.readthedocs.io/en/stable/Python-Intro.html
https://lightgbm.readthedocs.io/en/stable/Parameters.html
https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html
https://tdtapas.medium.com/sentiment-analysis-using-lightgbm-alternative-approach-to-rnn-and-lstm-55ee6f32e066

#### Tkinter
https://youtu.be/TuLxsvK4svQ?si=eYGamdDKLiIwHthd
https://youtu.be/YXPyB4XeYLA?si=gQsU7zzNNBKWA6xO
https://stackoverflow.com/questions/41371815/how-can-i-stop-my-tkinter-gui-from-freezing-when-i-click-my-button
