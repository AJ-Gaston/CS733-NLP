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

    #Fixed this line because of setting warnings
    df['publishedAtDate'] = pd.to_datetime(
        df['publishedAtDate'], 
        format='ISO8601'
    ).dt.tz_localize(None)
    
    df['text'] = df['text'].fillna('[NO_TEXT]')
    df['textTranslated'] = df['textTranslated'].fillna('[NO_TEXT]')
    
    df['Language'] = df['language'].fillna('unknown')
    
    return df

def analyze_restaurant_multiple_windows(model, restaurant_df, review_windows= [1,3,5,7,10]):
    """
    Analyze restaurant sentiment across multiple review windows
    Returns analysis for each window that has enough data
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
        'restaurant': restaurant_df['title'].iloc[0],
        'address': restaurant_df['address'].iloc[0] if 'address' in restaurant_df.columns else '',
        'city': restaurant_df['city'].iloc[0] if 'city' in restaurant_df.columns else '',
        'totalScore': restaurant_df['totalScore'].iloc[0],
        'total_reviews': len(restaurant_df),
        'windows_analyzed': results,
        'historical_stars_avg': restaurant_df['stars'].mean(),
        'first_review_date': restaurant_df['publishedAtDate'].min(),
        'latest_review_date': restaurant_df['publishedAtDate'].max()
    }
    
def save_detailed_results(all_results):
    """Save multi-window analysis to CSV for further exploration"""
    rows = []
    
    for result in all_results:
        restaurant_info = {
            'restaurant': result['restaurant'],
            'city': result['city'],
            'totalScore': result['totalScore'],
            'total_reviews': result['total_reviews'],
            'historical_stars_avg': result['historical_stars_avg']
        }
        
        for window in result['windows_analyzed']:
            row = restaurant_info.copy()
            row.update({
                'window_size': window['window_size'],
                'recent_sentiment': window['recent_sentiment_mean'],
                'sentiment_gap': window['sentiment_gap'],
                'significant': window['significant_discrepancy'],
                'direction': window['direction'],
                'positive_ratio': window['positive_ratio'],
                'negative_ratio': window['negative_ratio'],
                'recent_stars_avg': window['recent_stars_avg']
            })
            rows.append(row)
    
    results_df = pd.DataFrame(rows)
    results_df.to_csv('multi_window_analysis.csv', index=False)
    print(f"Saved {len(results_df)} rows to 'multi_window_analysis.csv'")
    
    # Also create summary by restaurant
    summary_rows = []
    for result in all_results:
        if result['windows_analyzed']:
            # Use the largest available window for summary
            largest_window = result['windows_analyzed'][-1]
            summary_rows.append({
                'restaurant': result['restaurant'],
                'city': result['city'],
                'totalScore': result['totalScore'],
                'total_reviews': result['total_reviews'],
                'largest_window': largest_window['window_size'],
                'sentiment_gap': largest_window['sentiment_gap'],
                'direction': largest_window['direction'],
                'significant': largest_window['significant_discrepancy']
            })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv('restaurant_summary.csv', index=False)
    print(f"Saved {len(summary_df)} rows to 'restaurant_summary.csv'")

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
    
    # Analyze each restaurant with multiple windows
    print("\nAnalyzing restaurants with multiple review windows...")
    all_results = []
    windows = [1, 3, 5, 7, 10]
    
    for restaurant_name, restaurant_df in df.groupby('title'):
        analysis = analyze_restaurant_multiple_windows(
            individual_model, 
            restaurant_df, 
            review_windows=windows
        )
        
        if analysis:
            all_results.append(analysis)
    
    # Generate comprehensive insights
    print(f" Analyzed {len(all_results)} restaurants")
    
    # Count restaurants by number of windows they could support
    window_counts = {}
    for result in all_results:
        windows_available = len(result['windows_analyzed'])
        window_counts[windows_available] = window_counts.get(windows_available, 0) + 1
    
    print("\nRestaurants by available review windows:")
    for windows_count, restaurant_count in sorted(window_counts.items()):
        window_list = windows[:windows_count]
        print(f"  {windows_count} windows ({window_list}): {restaurant_count} restaurants")
    
    # Analyze trends across window sizes
    for window_size in windows:
        # Get results for this window size (where available)
        window_results = []
        for result in all_results:
            for window in result['windows_analyzed']:
                if window['window_size'] == window_size:
                    window_results.append({
                        'restaurant': result['restaurant'],
                        'gap': window['sentiment_gap'],
                        'significant': window['significant_discrepancy'],
                        'direction': window['direction']
                    })
        #Should print out results if the window results was created
        if window_results:
            # Sentiment gap
            gaps = [r['gap'] for r in window_results]
            # Look at the significant discrepancies in the reviews and totalScore
            significant_count = sum(r['significant'] for r in window_results)
            # Look at where the restaurant is improving/declining
            improving_count = sum(1 for r in window_results if r['direction'] == 'improving')
            declining_count = sum(1 for r in window_results if r['direction'] == 'declining')
            
            print(f"\nWindow Size: {window_size} reviews")
            print(f"  Restaurants analyzed: {len(window_results)}")
            print(f"  Average gap: {np.mean(gaps):.3f}")
            print(f"  Significant discrepancies: {significant_count} ({significant_count/len(window_results):.1%})")
            print(f"  Improving: {improving_count}, Declining: {declining_count}, Stable: {len(window_results)-improving_count-declining_count}")
            
    consistent_improving = []
    consistent_declining = []

    for result in all_results:
        if len(result['windows_analyzed']) >= 3:  # At least 3 windows analyzed
            directions = [w['direction'] for w in result['windows_analyzed']]
            gaps = [w['sentiment_gap'] for w in result['windows_analyzed']]
        
            # Check if all windows show same trend
            if all(d == 'improving' for d in directions) and all(g > 0 for g in gaps):
                consistent_improving.append(result)
            elif all(d == 'declining' for d in directions) and all(g < 0 for g in gaps):
                consistent_declining.append(result)
    
    # Display the consistently improving and declining resaturants
    print(f"\nConsistently Improving: {len(consistent_improving)} restaurants")
    print(f"Consistently Declining: {len(consistent_declining)} restaurants")
    
    #Print out the top 5 consistently improving
    if consistent_improving:
        print("\nTop 5 Most Consistently Improving Restaurants:")
        for i, result in enumerate(sorted(consistent_improving, 
                                        key=lambda x: np.mean([w['sentiment_gap'] for w in x['windows_analyzed']]), 
                                        reverse=True)[:5], 1):
            avg_gap = np.mean([w['sentiment_gap'] for w in result['windows_analyzed']])
            print(f"{i}. {result['restaurant']} - Avg Gap: +{avg_gap:.3f}")
            for window in result['windows_analyzed']:
                print(f"   Window {window['window_size']}: {window['sentiment_gap']:+.3f} ({window['direction']})")
                
    print("\n Saving detailed analysis...")
    save_detailed_results(all_results)       
    return all_results
       
if __name__ == '__main__':
    main()