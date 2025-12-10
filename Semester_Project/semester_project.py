#import the libraries
import pandas as pd
from restaurant_sentiment_analysis import RestaurantSentimentAnalysisModel

def prepare_dataset(df):
    def star_to_sentiment(star):
        """ 
        A nested function that converts star ratings to sentiment
        Args:
            star (int): review rating of a restaurant

        Returns:
            int: sentiment of star rating
        """
        if pd.isna(star):
            return 1  # Default to neutral if missing
        if star <= 2:
            return 0  # negative
        if star == 3:
            return 1  # neutral
        if star >= 4:
            return 2  # positive
        
    df['sentiment'] = df['stars'].apply(star_to_sentiment)
    
    # 2. Handle dates
    df['publishedAtDate'] = pd.to_datetime(df['publishedAtDate'])
    
    #Fill in missing reviews
    df['text'] = df['text'].fillna('[NO_TEXT]')
    df['translatedText'] = df['translatedText'].fillna('[NO_TEXT]')
    
    # 4. Language encoding (for metadata features, not text)
    df['Language'] = df['language'].fillna('unknown')
    
    return df

def analyze_restaurant_trend(restaurant_df, top_n=10):
    """

    Args:
        restaurant_df (pd.Dataframe): dataframe of a specific restaurant
        top_n (int, optional): top n reviews. Defaults to 10.

    Returns:
        dict: a dictionary of 
    """
    if len(restaurant_df) < top_n:
        return None
    restaurant_df = restaurant_df.sort_values('publishedAtDate', ascending=False)
    recent_reviews = restaurant_df.head(top_n)
    
    # Get restaurant's totalScore
    total_score = restaurant_df['totalScore'].iloc[0] / 5.0  # Normalize to 0-1
    
    return

def main():
    reviews = pd.read_csv('./Dataset/Filtered_GoogleMaps_reviews.csv') #read in dataset with pandas
    #split each restaurant into their own dataframe
    print(f"Dataset shape: {reviews.shape}")
    print(f"Columns: {list(reviews.columns)}")
    print(f"Sample restaurant: {reviews['title'].iloc[0]}")
    
    df = prepare_dataset(reviews)
    
    print(f"After preprocessing:")
    print(f"- Total reviews: {len(df)}")
    print(f"- Unique restaurants: {df['restaurant'].nunique()}")
    print(f"- Date range: {df['publishedAtDate'].min()} to {df['publishedAtDate'].max()}")
    print(f"- Sentiment distribution: {dict(df['sentiment'].value_counts().sort_index())}")
    
    #Train the sentiment model
    individual_model = RestaurantSentimentAnalysisModel(use_restaurant_features=True)
    individual_model.train(df)
    
    # Show feature importance
    print("\nTop 10 most important features:")
    for feature, importance in individual_model.get_feature_importance(top_n=10):
        print(f"  {feature}: {importance:.4f}")
    
    results = []
    restaurant_count = 0
    for restaurant_name, restaurant_df in df.groupby('restaurant'):
        restaurant_count += 1
        print(f"Processed {restaurant_count} restaurants...")
    
    
    #Save Results for further analysis


if __name__ == '__main__':
    main()