#import the libraries
import pandas as pd
import numpy as np
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

    df['publishedAtDate'] = pd.to_datetime(df['publishedAtDate'], format='ISO8601')
    df['publishedAtDate'] = df['publishedAtDate'].dt.tz_localize(None)  # Remove timezone   
    
    df['text'] = df['text'].fillna('[NO_TEXT]')
    df['textTranslated'] = df['textTranslated'].fillna('[NO_TEXT]')
    
    df['Language'] = df['language'].fillna('unknown')
    
    return df

def analyze_restaurant_trends(model, restaurant_df, review_windows= [1,3,5,7,10]):
    """_summary_

    Args:
        model (RestaurantSentimentModel): an instance of the sentiment analaysis model
        restaurant_df (pd Dataframe): dataframe of a sepcific restaurant
        review_windows (list, optional): list of top n reviews to analyze. Defaults to [1,3,5,7,10].

    Returns:
        dict: a dictionary containing the window review size, sentiment analysis, and other analysis
    """
    if len(restaurant_df) == 0:
        return None
    
    # Sort by date (most recent first)
    restaurant_df = restaurant_df.sort_values('publishedAtDate', ascending=False)
    
    results = []
    restaurant_total_score = restaurant_df['totalScore'].iloc[0] / 5.0
    
    for window_size in review_windows:
        # Skip if not enough reviews for this window
        if len(restaurant_df) < window_size:
            continue
            
        recent_reviews = restaurant_df.head(window_size)
        
        # Predict sentiment based on the the top n reviews in the windows
        recent_predictions = model.predict(recent_reviews, return_prob=True)
        
        # Calculate sentiment scores (positive - negative)
        sentiment_scores = recent_predictions[:, 2] - recent_predictions[:, 0]
        mean_sentiment = np.mean(sentiment_scores)

        sentiment_gap = mean_sentiment - restaurant_total_score
        
        # Dynamic significance threshold based on window size
        # Smaller windows need larger gaps to be significant
        if window_size <= 3: 
            sig_threshold = 0.25  # More conservative for small windows
        elif window_size <= 7:
            sig_threshold = 0.20
        else:
            sig_threshold = 0.15
            
        significant = abs(sentiment_gap) > sig_threshold
        
        # Determine trend direction
        if sentiment_gap > 0.1:
            direction = 'improving'
        elif sentiment_gap < -0.1:
            direction = 'declining'
        else:
            direction = 'stable'
            
        window_result = {
            'window_size': window_size,
            'recent_sentiment_mean': mean_sentiment,
            'sentiment_gap': sentiment_gap,
            'significant_discrepancy': significant,
            'direction': direction,
            'positive_ratio': np.mean(sentiment_scores > 0.2),
            'negative_ratio': np.mean(sentiment_scores < -0.2),
            'recent_stars_avg': recent_reviews['stars'].mean(),
            'review_count': len(recent_reviews),
            'sample_texts': recent_reviews['text'].head(2).tolist() if window_size >= 3 else []
        }
        
        results.append(window_result)
    
    # Only return if we have at least one window analyzed
    if not results:
        return None
    
    #Otherise return the restaurant with review(s) sentiment analysis
    return {
        'restaurant': restaurant_df['restaurant'].iloc[0],
        'address': restaurant_df['address'].iloc[0] if 'address' in restaurant_df.columns else '',
        'city': restaurant_df['city'].iloc[0] if 'city' in restaurant_df.columns else '',
        'totalScore': restaurant_df['totalScore'].iloc[0],
        'total_reviews': len(restaurant_df),
        'windows_analyzed': results,
        'historical_stars_avg': restaurant_df['stars'].mean(),
        'first_review_date': restaurant_df['publishedAtDate'].min(),
        'latest_review_date': restaurant_df['publishedAtDate'].max()
    }

def main():
    reviews = pd.read_csv('./Dataset/Filtered_GoogleMaps_reviews.csv') #read in dataset with pandas
    #split each restaurant into their own dataframe
    print(f"Dataset shape: {reviews.shape}")
    print(f"Columns: {list(reviews.columns)}")
    print(f"Sample restaurant: {reviews['title'].iloc[0]}")
    
    df = prepare_dataset(reviews)
    
    print(f"After preprocessing:")
    print(f"- Total reviews: {len(df)}")
    print(f"- Unique restaurants: {df['title'].nunique()}")
    print(f"- Date range: {df['publishedAtDate'].min()} to {df['publishedAtDate'].max()}")
    print(f"- Sentiment distribution: {dict(df['sentiment'].value_counts().sort_index())}")
    
    #Train the sentiment model
    individual_model = RestaurantSentimentAnalysisModel(use_restaurant_features=True)
    individual_model.train_model(df)
    
    # Show feature importance
    print("\nTop 10 most important features:")
    for feature, importance in individual_model.get_feature_importance(top_n=10):
        print(f"  {feature}: {importance:.4f}")
    
    results = []
    restaurant_count = 0
    for restaurant_name, restaurant_df in df.groupby('title'):
        restaurant_count += 1
        print(f"Processed {restaurant_count} restaurants...")
    
    if len(restaurant_df) >= 10:  # Need enough reviews
            analysis = analyze_restaurant_trends(individual_model, restaurant_df, top_n=10)
            if analysis:
                results.append(analysis)
                
    print(f" Analyzed {len(results)} restaurants (with over 10 reviews)")
    
    # Find restaurants with significant discrepancies
    alerts = [r for r in results if r['significant_discrepancy']]
    improving = [r for r in alerts if r['direction'] == 'improving']
    declining = [r for r in alerts if r['direction'] == 'declining']
    
    print(f" Found {len(alerts)} restaurants with significant changes:")
    print(f" {len(improving)} improving (recent reviews better than average)")
    print(f" {len(declining)} declining (recent reviews worse than average)")
    
    #Look at the positive changes (sort them)
    positive_changes = sorted([r for r in results if r['sentiment_gap'] > 0], 
                            key=lambda x: x['sentiment_gap'], reverse=True)[:5]
    
    for i, r in enumerate(positive_changes, 1):
        print(f"\n{i}. {r['restaurant']}")
        print(f"   Location: {r['city']}")
        print(f"   Total Score: {r['totalScore']:.1f}/5")
        print(f"   Recent Sentiment: {r['recent_sentiment_mean']:.2f}")
        print(f"   Gap: +{r['sentiment_gap']:.3f} (↑ {r['direction']})")
        print(f"   Recent Stars Avg: {r['recent_stars_avg']:.1f}/5")
        print(f"   Historical Stars Avg: {r['historical_stars_avg']:.1f}/5")
    
    #Look at the negative changes (sort them)
    negative_changes = sorted([r for r in results if r['sentiment_gap'] < 0], 
                            key=lambda x: x['sentiment_gap'])[:5]
    
    for i, r in enumerate(negative_changes, 1):
        print(f"\n{i}. {r['restaurant']}")
        print(f"   Location: {r['city']}")
        print(f"   Total Score: {r['totalScore']:.1f}/5")
        print(f"   Recent Sentiment: {r['recent_sentiment_mean']:.2f}")
        print(f"   Gap: {r['sentiment_gap']:.3f} (↓ {r['direction']})")
        print(f"   Recent Stars Avg: {r['recent_stars_avg']:.1f}/5")
        print(f"   Historical Stars Avg: {r['historical_stars_avg']:.1f}/5")
    
    # Save results to CSV for further analysis
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        # Save with useful columns
        output_cols = ['restaurant', 'city', 'totalScore', 'recent_sentiment_mean', 
                      'sentiment_gap', 'significant_discrepancy', 'direction',
                      'recent_review_count', 'positive_ratio', 'negative_ratio',
                      'recent_stars_avg', 'historical_stars_avg']
        
        results_df[output_cols].to_csv('restaurant_sentiment_analysis.csv', index=False)
        print(f"Results saved to 'restaurant_sentiment_analysis.csv'")
    
    return results

if __name__ == '__main__':
    main()