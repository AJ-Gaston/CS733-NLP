import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import transformers
from transformers import DistilBertModel, DistilBertTokenizer
import lightgbm as lgb
import sklearn
from sklearn.model_selection import train_test_split
from datetime import datetime
import warnings

class RestaurantSentimentAnalysisModel:
    """
    Sentiment Analysis Model for restaurant reviews
    Handles: text (multilingual), ratings, dates, missing data
    """
    def __init__(self, text_model_name='distilbert-base-multilingual-cased',
                 use_restaurant_features=True,
                 use_caching=False,
                 cache_size=1000):
        
        #Text Encoding Components
        self.text_encoder = DistilBertModel.from_pretrained(text_model_name)
        self.tokenizer = DistilBertTokenizer.from_pretrained(text_model_name)
        
        # LightGBM model (will be trained)
        self.lgb_model = None
        
        self.feature_names = []
        
        self.use_restaurant_features = use_restaurant_features
        self.use_caching = use_caching
        self.cache = {} if use_caching else None
        self.cache_size = cache_size
        
        # Freeze DistilBERT (only use it for embeddings, not fine-tuning)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
    
    def _get_text_embeddings(self, texts, batch_size=32):
        """Extract embeddings from text using DistilBERT"""
        if not texts:
            return np.zeros((0, 768))
        embeddings = []
        
        #Process the text embedding in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            #Tokenize the batch using DistilBERT tokenizer
            inputs = self.tokenizer(
            batch_texts,
            padding=True,      # Pad short texts
            truncation=True,   # Truncate long texts  
            max_length=128,    # To 128 tokens max
            return_tensors='pt'
            )
            
            # Get DistilBERT embeddings
            with torch.no_grad():  # No gradient needed (inference only)
                outputs = self.text_encoder(**inputs)
                # Extract [CLS] token (first token, represents whole sentence)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
                embeddings.append(batch_embeddings)
        
        # Combine all batches using numpy's vstack() if the embeddings exist
        # Else create an empty numpy array  of shape (the length of texts, 768)
        return np.vstack(embeddings) if embeddings else np.zeros((len(texts),768))
    
    def _extract_temporal_features(self, df):
        """
        Extract features from the dataframe
        The review text(translated if it's not in English), category, star rating
        Args:
            df (pd Series): the dataframe containing the reviews
            batch_size (int, optional): number of batches for the model. Defaults to 32.

        Returns:
            dictionary: contians all the features of a restaurant's dataframe
        """
        #Ensure that the publishedAtDate is datetime format
        df['publishedAtDate'] = pd.to_datetime(df['publishedAtDate'], errors='coerce')
        #Look at the year, month, and day, normalize them to between 0-1 and reshape them to a column vector
        current_year = datetime.now().year #Get the current year (2025)
        year_norm = (df['publishedAtDate'].dt.year / current_year).fillna(0.5).values
        
        #Month -cyclic encoding (use 2*pi)
        month_rad = 2 * np.pi * df['publishedAtDate'].dt.month.fillna(6) / 12
        #Get the sin and cosine for cyclical distance
        month_sin = np.sin(month_rad).values
        month_cos = np.cos(month_rad).values
        
        #Day of the week and weekend
        day_of_week = (df['publishedAtDate'].dt.dayofweek.fillna(3) / 6.0).values
        is_weekend = df['publishedAtDate'].dt.dayofweek.isin([5, 6]).fillna(False).astype(int).values #check to see if it was a weekend
        
        #Return the temporal features using np.column_stack
        return np.column_stack([year_norm, month_sin, month_cos, day_of_week, is_weekend])
    
    def _create_feature_names(self, total_features, embedding_dim):
        """
        Creates descriptive feature names (Good for interpretability)
        """
        self.feature_names = []
        
        #Text Embedding Names
        self.feature_names.extend([f'text_emb_{i}' for i in range(embedding_dim)])
        
        #Core features (star rating, total socre rating, rating discrepancy)
        core_names = ['star_norm', 'total_score_norm', 'rating_discrepancy']
        self.feature_names.extend(core_names)
        
        # Temporal features
        temporal_names = ['year_norm', 'month_sin', 'month_cos', 'day_of_week', 'is_weekend']
        self.feature_names.extend(temporal_names)
        
        # Text metadata
        meta_names = ['has_original', 'has_translation', 'is_english', 
                     'text_length', 'word_count', 'text_source']
        self.feature_names.extend(meta_names)
        
        # Restaurant features
        if self.use_restaurant_features:
            restaurant_names = ['recency', 'has_exclamation', 'has_question']
            self.feature_names.extend(restaurant_names)
        
        # Ensure length matches
        if len(self.feature_names) != total_features:
            # Fill remaining with generic names
            for i in range(len(self.feature_names), total_features):
                self.feature_names.append(f'feature_{i}')
                
    def extract_features(self, df):
        """
        Extract all features from restaurant review dataframe
        df should contain: text, translatedLanguage, star, totalScore, 
                          publishedAtDate, originalLanguage, and title
        Args:
            df (pd Dataframe): a pandas Dataframe of a restaurant and its reviews

        Returns:
            dictionary: a dictionary containing all the features of a restaurant
        """
        texts_to_embed = []
        text_source_flags = []
        
        for _, row in df.iterrows():
            #Use translation if available and original is not English
            if (pd.notna(row.get('translatedLanguage')) and 
                str(row.get('originalLanguage', '')).lower() != 'en'):
                texts_to_embed.append(row['translatedLanguage'])
                text_source_flags.append(0)  # Translated
            elif pd.notna(row.get('text')):
                texts_to_embed.append(row['text'])
                text_source_flags.append(1)  # Get the original review
            else:
                texts_to_embed.append('[NO TEXT]')
                text_source_flags.append(2)  # Missing review text
        
        # Text embeddings of the reviews
        text_embeddings = self._get_text_embeddings(texts_to_embed)
        
        #Forgot to reshape the text_embeddings (cause an issue with np.hstack())
        text_source_flags_array = np.array(text_source_flags).reshape(-1, 1)
        
        # Numerical Features (review star rating and restaurant's total score rating
        star_norm = (df['stars'] / 5.0).fillna(0.5).values.reshape(-1, 1)
        total_score_norm = (df['totalScore'] / 5.0).fillna(0.5).values.reshape(-1, 1)
        
        # Rating discrepancy between the reviewer's and the restaurant
        rating_discrepancy = (df['totalScore'] - df['stars']).fillna(0).values.reshape(-1, 1) / 5.0
        
        # Temporal features
        temporal_features = self._extract_temporal_features(df)
        
        # Text Metadata (original review, translated review, and is it english)
        has_original = df['text'].notna().astype(int).values.reshape(-1, 1)
        has_translation = df['translatedLanguage'].notna().astype(int).values.reshape(-1, 1)
        is_english = df['originalLanguage'].fillna('').str.lower().eq('en').astype(int).values.reshape(-1, 1)
        
        #Text quality features (length of review and word count)
        text_length = pd.Series(texts_to_embed).str.len().fillna(0).values.reshape(-1, 1)
        word_count = pd.Series(texts_to_embed).str.split().str.len().fillna(0).values.reshape(-1, 1)
        
        #Restaurant's specific features
        restaurant_features = np.empty((len(df), 0)) #create an empty ndarray [df length,0]
        if self.use_restaurant_features:
            # Review recency (inverse of days ago)
            if 'publishedAtDate' in df.columns:
                df['publishedAtDate'] = pd.to_datetime(df['publishedAtDate'], errors='coerce')
                days_ago = (pd.Timestamp.now() - df['publishedAtDate']).dt.days.fillna(365)
                recency = 1 / (1 + days_ago / 30)  # Decay over months
                recency = recency.values.reshape(-1, 1)
            else:
                recency = np.ones((len(df), 1))
            
            # Emotional intensity indicators
            texts_series = pd.Series(texts_to_embed)
            has_exclamation = texts_series.str.contains('!').astype(int).values.reshape(-1, 1)
            has_question = texts_series.str.contains(r'\?').astype(int).values.reshape(-1, 1)
            
            restaurant_features = np.column_stack([recency, has_exclamation, has_question])
        # Combine all features
        all_features = np.hstack([
            text_embeddings,            # Text semantics (768 dim)
            star_norm,                  # Individual rating
            total_score_norm,           # Restaurant average score
            rating_discrepancy,         # IMPORTANT!!!!
            temporal_features,          # Date features (5 dim)
            has_original,               # Text presence flags
            has_translation,
            is_english,
            text_length,                # Text quality
            word_count,
            text_source_flags_array,    # 0=translated, 1=original, 2=missing (Wasn't the right shape originally)
            restaurant_features         # Optional restaurant features
        ])
        
        # Create feature names for interpretability 
        self._create_feature_names(all_features.shape[1], text_embeddings.shape[1])
        return all_features

    def train_model(self, df, test_size=0.2, random_state=42):
        """
        Train the model based on the review
        (text, original(if available), and text translated(if it's available with original)) 
        and star associated with review
        
        Args:
            df (pd Dataframe): a pandas Dataframe of a restaurant and its reviews
            test_size (float or int, optional): the size of dataset split
            random_state (int, optional): the random seed for splitting dataframe

        Returns:
            dictionary: a dictionary containing all the features of a restaurant
        
        """
        # Create labels from star ratings if needed
        if 'sentiment' not in df.columns:
            # Convert stars to sentiment classes
            # 1-2 stars: negative (0), 3 stars: neutral (1), 4-5 stars: positive (2)
            df['sentiment'] = pd.cut(df['stars'], 
                                     bins=[0, 2, 3, 5], 
                                     labels=[0, 1, 2], 
                                     right=True).astype(int)
        
        # Extract features
        X = self.extract_features(df)
        y = df['sentiment'].values
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # LightGBM parameters optimized for sentiment analysis
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 4,
            'max_depth': 7,
            'min_data_in_leaf': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }
        
        # Train LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        self.lgb_model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30),
                lgb.log_evaluation(period=100)
            ]
        )
        
        # Evaluate
        train_pred = np.argmax(self.lgb_model.predict(X_train), axis=1)
        val_pred = np.argmax(self.lgb_model.predict(X_val), axis=1)
        
        train_acc = np.mean(train_pred == y_train)
        val_acc = np.mean(val_pred == y_val)
        
        print(f"Training accuracy: {train_acc:.3f}")
        print(f"Validation accuracy: {val_acc:.3f}")
        
        return self.lgb_model

    def predict(self,df, return_prob=False):
        """
        Predicts the sentiment of new reviews
        Args:
            df (pd Dataframe): 
            return_prob (bool, optional): indicates whether or not to return the probabilities. Defaults to False.
        """
        #Check to see if the model was trained
        if self.lgb_model is None:
            raise ValueError("Model must be trained before prediction")
        
        # Extract features
        X = self.extract_features(df)
        
        # Check cache if enabled
        if self.use_caching and self.cache is not None:
            cache_key = self._create_cache_key(df)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        # Make predictions
        if return_prob:
            predictions = self.lgb_model.predict(X)
        else:
            predictions = np.argmax(self.lgb_model.predict(X), axis=1)
        
        # Store in cache if enabled
        if self.use_caching and self.cache is not None:
            self.cache[cache_key] = predictions
            # Manage cache size
            if len(self.cache) > self.cache_size:
                # Remove oldest entry
                self.cache.pop(next(iter(self.cache)))
        
        return predictions

    def _create_cache_key(self, df):
        """Create cache key from dataframe content"""
        # Use hash of concatenated text and ratings
        texts = df['text'].fillna('') + df['translatedLanguage'].fillna('')
        ratings = df['stars'].astype(str) + df['totalScore'].astype(str)
        key_str = ''.join(texts) + ''.join(ratings)
        return hash(key_str)
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance for interpretability"""
        if self.lgb_model is None:
            raise ValueError("Model not trained")
        
        importance = self.lgb_model.feature_importance(importance_type='gain')
        
        # Create sorted list
        features_with_importance = list(zip(self.feature_names, importance))
        features_with_importance.sort(key=lambda x: x[1], reverse=True)
        
        return features_with_importance[:top_n]