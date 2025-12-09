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
       
        #Numerical Features (specifically RATINGS)
        # Core ratings (normalized)
        star_norm = (df['stars'] / 5.0).values.reshape(-1, 1)
        total_score_norm = (df['totalScore'] / 5.0).values.reshape(-1, 1)
        
        # MOST IMPORTANT: Rating discrepancy (the actual review - the customer's review)
        discrepancy = ((df['totalScore'] - df['stars']) / 5.0).values.reshape(-1, 1)
        
        #Time features
        df['publishedAtDate'] = pd.to_datetime(df['publishedAtDate']) #Convert to datetime if it's not already converted
        #Look at the year, month, and day, normalize them to between 0-1 and reshape them to a column vector
        year_norm = (df['publishedAtDate'].dt.year / 2024.0).values.reshape(-1, 1) #Get year
        month_norm = (df['publishedAtDate'].dt.month / 12.0).values.reshape(-1, 1)
        day_of_week = (df['publishedAtDate'].dt.dayofweek / 6.0).values.reshape(-1, 1)
        is_weekend = df['publishedAtDate'].dt.dayofweek.isin([5, 6]).astype(int).values.reshape(-1, 1) #check to see if it was a weekend
       
        #Language features
        df['language'] = df['language'].fillna('unknown')
        language_major = df['language'].isin(['en', 'es', 'fr', 'de', 'zh', 'kr', 'it']).astype(int).values.reshape(-1, 1)
        is_english = (df['language'] == 'en').astype(int).values.reshape(-1, 1)
        
        #Make a numerical feature matrix  using np.column_stack
        numerical_features = np.column_stack([
        star_norm,           # Individual rating
        total_score_norm,    # Restaurant average
        discrepancy,         # Difference (GOLDEN feature)
        year_norm,           # Year
        month_norm,          # Month
        day_of_week,         # Day of week
        is_weekend,          # Weekend flag
        language_major,      # Common language
        is_english           # English flag
        ])
        
        # Combine ALL features using np.hstack()
        all_features = np.hstack([
            original_embeddings,     # 768 dim: Original text semantics
            translated_embeddings,   # 768 dim: Translated text semantics
            has_original_text,           # 1 dim: Has original text
            has_translated_text,         # 1 dim: Has translation
            text_length,            # 1 dim: Character count
            word_count,             # 1 dim: Word count
            has_exclamation,        # 1 dim: Emotional intensity
            has_question,           # 1 dim: Question indicator
            numerical_features      # 9 dim: All numerical features
        ])
        
        #Create feature names just for checking
        self.feature_names = (
        [f'orig_emb_{i}' for i in range(768)] +
        [f'trans_emb_{i}' for i in range(768)] +
        ['has_original', 'has_translated', 'text_length', 'word_count', 
         'has_exclamation', 'has_question'] +
        ['star_norm', 'total_score_norm', 'rating_discrepancy', 'year_norm',
         'month_norm', 'day_of_week', 'is_weekend', 'language_major', 'is_english']
        )
       
        #This is used to see if everything works(will comment out later)
        print(f"Feature dimensions:")
        print(f"  Original embeddings: {original_embeddings.shape[1]}")
        print(f"  Translated embeddings: {translated_embeddings.shape[1]}")
        print(f"  Text quality: 6 features")
        print(f"  Numerical: {numerical_features.shape[1]} features")
        print(f"  TOTAL: {all_features.shape[1]} features")
        return all_features
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