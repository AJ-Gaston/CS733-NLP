import torch
import torch.nn as nn
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd

class RestaurantSentimentAnalysisModel:
    def __init__(self,model_name='distilbert-base-multilingual-cased'):
        self.text_model = DistilBertModel.from_pretrained(model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.lgb_model = None
        self.feature_names = None

    def _extract_features(self, df, batch_size=32):
        """
        Extract features from the dataframe
        The review text(translated if it's not in English), category, star rating
        Args:
            df (pd Series): the dataframe containing the reviews
            batch_size (int, optional): number of batches for the model. Defaults to 32.

        Returns:
            dictionary: contians all the features of a restaurant's dataframe
        """
        #TEXT EMBEDDINGS OF DATAFRAME
        # Get the original text embeddings
        original_texts = df['text'].fillna('[NO TEXT]').tolist()
        original_embeddings = self._get_text_embeddings(original_texts, batch_size)
        
        # Get the translated text embeddings
        translated_texts = df['translatedLanguage'].fillna('[NO TRANSLATION]').tolist()
        translated_embeddings = self._get_text_embeddings(translated_texts, batch_size)
        
        #TEXT QUALITY INDICATORS OF REVIEWS
        has_original_text = df['text'].notna().astype(int).values.reshape(-1, 1)
        has_translated_text = df['translatedLanguage'].notna().astype(int).values.reshape(-1, 1)
        
        # Text length (more sophisticated)
        combined_text = df['translatedLanguage'].fillna(df['text'])
        #Reshape the text length and word count of the combined text
        text_length = combined_text.str.len().fillna(0).values.reshape(-1, 1)
        word_count = combined_text.str.split().str.len().fillna(0).values.reshape(-1, 1)
        
        # Emotional intensity indicators (Exclamation point and quesiton marks are a good indicator)
        has_exclamation = combined_text.str.contains('!').astype(int).values.reshape(-1, 1)
        has_question = combined_text.str.contains(r'\?').astype(int).values.reshape(-1, 1)
        
        #Time features
        df['publishedAtDate'] = pd.to_datetime(df['publishedAtDate']) #Convert to datetime if it's not already converted
        #Look at the year, month, and day, normalize them to between 0-1 and reshape them to a column vector
        year_norm = (df['publishedAtDate'].dt.year / 2024.0).values.reshape(-1, 1) #Get year
        month_norm = (df['publishedAtDate'].dt.month / 12.0).values.reshape(-1, 1)
        day_of_week = (df['publishedAtDate'].dt.dayofweek / 6.0).values.reshape(-1, 1)
        
        
        return features
    def _get_text_embeddings(self, texts, batch_size=32):
        """Get CLS token embeddings from DistilBERT"""
        embeddings = []
        return embeddings 
    def forward():
        """
        Forward pass of the model

        Returns:
            _type_: _description_
        """
        return output
    def train_model(model, trainloader):
        """
        Train the model based on the review
        (text, original(if available), and text translated(if it's available with original)) 
        and star associated with review
        """
        return
    def evaluate_model(y_true,y_pred):
        """
        This evaluates the model's accuracy, precision, recall, and f1-score
        Also used to compare the top 10 reviews to the restaurant's overall score
        """
        return