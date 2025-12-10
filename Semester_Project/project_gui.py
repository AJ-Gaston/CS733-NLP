from tkinter import *
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import os
import threading
import semester_project
import restaurant_sentiment_analysis

class RestaurantSentimentGUI:
    """
    Class that creates the GUI for restaurant sentiment
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Restaurant Sentiment Analyzer")
        self.root.geometry("800x600")
        
        #Initialize the data structures
        self.df = None
        self.model = None
        self.restaurant_data = None
        self.results = None
        
        #Review window options
        self.review_windows = [1,3,5,7,10]
        
         # Add a loading flag for threading
        self.is_loading = False
        self.loading_thread = None
            
        #Configure the styles
        self.setup_styles()
        
        #Build the GUI
        self.setup_gui()
        
        #Load data in the background
        self.load_data()
        
    def setup_styles(self):
        """
        Configure the colors and fonts for the GUI
        """
        self.colors = {
            'positive': '#4CAF50',      # Green for improvement
            'negative': '#F44336',      # Red for decline
            'neutral': '#2196F3',       # Blue for stable
            'background': '#f5f5f5',    # Light gray background
            'card': '#ffffff',          # White for cards
            'text': '#212121'           # Dark gray text
        }
        
        self.fonts = {
            'title': ('Arial', 16, 'bold'),
            'heading': ('Arial', 12, 'bold'),
            'normal': ('Arial', 10),
            'metric': ('Arial', 14, 'bold'),
            'status': ('Arial', 10, 'bold')
        }
        
    def setup_gui(self):
        """Build the GUI interface"""
        # Set background color
        self.root.configure(bg=self.colors['background'])
        
        # Main container
        main_container = Frame(self.root, bg=self.colors['background'])
        main_container.pack(fill=BOTH, expand=True, padx=20, pady=20)
        
        # Title Section
        title_frame = Frame(main_container, bg=self.colors['background'])
        title_frame.pack(fill=X, pady=(0, 20))
        
        title_label = Label(title_frame, text=" Restaurant Sentiment Analyzer", 
                          font=self.fonts['title'], bg=self.colors['background'], 
                          fg=self.colors['text'])
        title_label.pack()
        
        subtitle_label = Label(title_frame, 
                             text="Analyze recent review sentiment vs. overall ratings",
                             font=self.fonts['heading'], bg=self.colors['background'], 
                             fg='#666666')
        subtitle_label.pack()
        
        # Control Panel
        control_frame = Frame(main_container, bg=self.colors['card'], 
                            relief=GROOVE, bd=2, padx=15, pady=15)
        control_frame.pack(fill=X, pady=(0, 20))
        
        # Restaurant Selection
        Label(control_frame, text="Restaurant:", font=self.fonts['heading'], 
              bg=self.colors['card']).grid(row=0, column=0, sticky=W, padx=(0, 10), pady=5)
        
        self.restaurant_var = StringVar()
        self.restaurant_dropdown = ttk.Combobox(control_frame, textvariable=self.restaurant_var, 
                                                width=35, state="readonly", font=self.fonts['normal'])
        self.restaurant_dropdown.grid(row=0, column=1, padx=(0, 20), pady=5, sticky=W)
        self.restaurant_dropdown.bind('<<ComboboxSelected>>', self.on_restaurant_selected)
        
        # Review Window Selection
        Label(control_frame, text="Review Window:", font=self.fonts['heading'], 
              bg=self.colors['card']).grid(row=0, column=2, sticky=W, padx=(0, 10), pady=5)
        
        self.window_var = IntVar()
        self.window_dropdown = ttk.Combobox(control_frame, textvariable=self.window_var, 
                                            width=12, state="readonly", font=self.fonts['normal'])
        self.window_dropdown.grid(row=0, column=3, padx=(0, 20), pady=5, sticky=W)
        self.window_dropdown.bind('<<ComboboxSelected>>', self.on_window_selected)
        
        # Analyze Button
        self.analyze_btn = Button(control_frame, text=" Analyze", command=self.analyze_restaurant,
                                 font=self.fonts['heading'], bg=self.colors['neutral'], 
                                 fg='white', padx=20, state=DISABLED, cursor="hand2")
        self.analyze_btn.grid(row=0, column=4, padx=(10, 0), pady=5)
        
        # Results Display Area
        results_frame = Frame(main_container, bg=self.colors['background'])
        results_frame.pack(fill=BOTH, expand=True)
        
        # Restaurant Info Card
        info_card = Frame(results_frame, bg=self.colors['card'], relief=RAISED, bd=1, padx=15, pady=10)
        info_card.pack(fill=X, pady=(0, 15))
        
        self.restaurant_name_label = Label(info_card, text="No restaurant selected", 
                                          font=self.fonts['heading'], bg=self.colors['card'])
        self.restaurant_name_label.pack(anchor=W)
        
        info_subframe = Frame(info_card, bg=self.colors['card'])
        info_subframe.pack(fill=X, pady=(5, 0))
        
        self.review_count_label = Label(info_subframe, text="Total reviews: --", 
                                       font=self.fonts['normal'], bg=self.colors['card'], fg='#666666')
        self.review_count_label.pack(side=LEFT, padx=(0, 20))
        
        self.date_range_label = Label(info_subframe, text="Date range: --", 
                                     font=self.fonts['normal'], bg=self.colors['card'], fg='#666666')
        self.date_range_label.pack(side=LEFT)
        
        # Metrics Display Card
        metrics_card = Frame(results_frame, bg=self.colors['card'], relief=RAISED, bd=1, padx=20, pady=20)
        metrics_card.pack(fill=BOTH, expand=True)
        
        # Create grid for metrics
        metrics_grid = Frame(metrics_card, bg=self.colors['card'])
        metrics_grid.pack(expand=True)
        
        # Row 0: Total Score and Recent Sentiment
        total_frame = Frame(metrics_grid, bg=self.colors['card'])
        total_frame.grid(row=0, column=0, sticky=W, pady=(0, 15))
        
        Label(total_frame, text="Total Score", font=self.fonts['normal'], 
              bg=self.colors['card']).pack(anchor=W)
        self.total_score_value = Label(total_frame, text="--", font=self.fonts['metric'], 
                                      bg=self.colors['card'])
        self.total_score_value.pack(anchor=W)
        
        recent_frame = Frame(metrics_grid, bg=self.colors['card'])
        recent_frame.grid(row=0, column=1, sticky=W, padx=(40, 0), pady=(0, 15))
        
        Label(recent_frame, text="Recent Sentiment", font=self.fonts['normal'], 
              bg=self.colors['card']).pack(anchor=W)
        self.recent_sentiment_value = Label(recent_frame, text="--", font=self.fonts['metric'], 
                                           bg=self.colors['card'])
        self.recent_sentiment_value.pack(anchor=W)
        
        # Row 1: Sentiment Gap
        gap_frame = Frame(metrics_grid, bg=self.colors['card'])
        gap_frame.grid(row=1, column=0, sticky=W, pady=(0, 15))
        
        Label(gap_frame, text="Sentiment Gap", font=self.fonts['normal'], 
              bg=self.colors['card']).pack(anchor=W)
        gap_subframe = Frame(gap_frame, bg=self.colors['card'])
        gap_subframe.pack(anchor=W)
        
        self.gap_value = Label(gap_subframe, text="--", font=self.fonts['metric'], 
                              bg=self.colors['card'])
        self.gap_value.pack(side=LEFT)
        self.gap_icon = Label(gap_subframe, text="", font=self.fonts['metric'], 
                             bg=self.colors['card'])
        self.gap_icon.pack(side=LEFT, padx=(5, 0))
        
        # Row 1: Direction
        direction_frame = Frame(metrics_grid, bg=self.colors['card'])
        direction_frame.grid(row=1, column=1, sticky=W, padx=(40, 0), pady=(0, 15))
        
        Label(direction_frame, text="Trend Direction", font=self.fonts['normal'], 
              bg=self.colors['card']).pack(anchor=W)
        self.direction_value = Label(direction_frame, text="--", font=self.fonts['metric'], 
                                    bg=self.colors['card'])
        self.direction_value.pack(anchor=W)
        
        # Additional Info Section
        additional_frame = Frame(metrics_card, bg=self.colors['card'])
        additional_frame.pack(fill=X, pady=(20, 0))
        
        Label(additional_frame, text="Additional Statistics", font=self.fonts['heading'], 
              bg=self.colors['card']).pack(anchor=W, pady=(0, 10))
        
        self.additional_info_text = Text(additional_frame, height=6, font=self.fonts['normal'], 
                                        bg=self.colors['card'], relief=FLAT, wrap=WORD)
        self.additional_info_text.pack(fill=X)
        self.additional_info_text.insert(END, "Select a restaurant and review window to see detailed analysis.")
        self.additional_info_text.config(state=DISABLED)
        
        # Status Bar
        self.status_frame = Frame(self.root, bg='#e0e0e0', height=25)
        self.status_frame.pack(fill=X, side=BOTTOM)
        
        self.status_label = Label(self.status_frame, text="Initializing...", 
                                 font=self.fonts['status'], bg='#e0e0e0', fg='#666666', 
                                 anchor=W, padx=10)
        self.status_label.pack(fill=X)
        
    def load_data(self):
        """
        Load data and train the model in the background
        """
        if self.is_loading:
            return  # Prevent multiple loads
        
        self.is_loading = True
        self.status_label.config(text="Loading data... Please wait")
        
        # Disable controls during loading
        self.restaurant_dropdown.config(state=DISABLED)
        self.window_dropdown.config(state=DISABLED)
        self.analyze_btn.config(state=DISABLED)
        
        def load_thread():
            """Background thread for loading data"""
            try:
                csv_path = './Dataset/Filtered_GoogleMaps_reviews.csv'  # Update this path
                
                if not os.path.exists(csv_path):
                    self.root.after(0, lambda: self.show_error(f"File not found: {csv_path}"))
                    return
                
                # Load and prepare data
                reviews = pd.read_csv(csv_path)
                df_loaded = semester_project.prepare_dataset(reviews)
                # Update in main thread
                self.root.after(0, lambda: self.update_status("Training model..."))
            
                # Train model
                model = semester_project.RestaurantSentimentAnalysisModel(use_restaurant_features=True)
                model.train_model(df_loaded)
            
                # Get restaurants list
                restaurants = sorted(df_loaded['title'].unique())
                
                # Update GUI in main thread
                self.root.after(0, lambda: self.finish_loading(df_loaded, model, restaurants))
            
            except Exception as e:
                self.root.after(0, lambda: self.loading_failed(str(e)))
    
        # Start the background thread
        self.loading_thread = threading.Thread(target=load_thread, daemon=True)
        self.loading_thread.start()
        
    def update_status(self, message):
        """Update status label from thread"""
        self.status_label.config(text=message)
    
    def finish_loading(self, df_loaded, model, restaurants):
        """Called when loading completes successfully"""
        # Store data
        self.df = df_loaded
        self.model = model
        
        # Update restaurant dropdown
        self.restaurant_dropdown['values'] = restaurants
        if restaurants:
            self.restaurant_dropdown.set(restaurants[0])
            self.on_restaurant_selected()
        
        # Re-enable controls
        self.restaurant_dropdown.config(state="readonly")
        if hasattr(self, 'current_restaurant_df'):
            self.window_dropdown.config(state="readonly")
            self.analyze_btn.config(state=NORMAL, bg=self.colors['neutral'])
        
        # Update status and hide progress bar
        self.status_label.config(
            text=f" Ready! {len(restaurants)} restaurants loaded")
        
        self.is_loading = False
    
    def loading_failed(self, error_msg):
        """Handle loading failure"""
        messagebox.showerror("Loading Error", f"Failed to load data: {error_msg}")
        self.status_label.config(text=" Failed to load data")
        
        # Re-enable at least restaurant dropdown for retry
        self.restaurant_dropdown.config(state="readonly")
        self.is_loading = False
        
    def update_restaurant_list(self, restaurants):
        self.restaurant_dropdown['values'] = restaurants
        if restaurants:
            self.restaurant_dropdown.set(restaurants[0])
            self.on_restaurant_selected()
            
    def clear_results_display(self):
        """Clear the results display area"""
        self.total_score_value.config(text="--")
        self.recent_sentiment_value.config(text="--")
        self.gap_value.config(text="--")
        self.gap_icon.config(text="")
        self.direction_value.config(text="--")
        
        # Clear additional info
        self.additional_info_text.config(state=NORMAL)
        self.additional_info_text.delete(1.0, END)
        self.additional_info_text.insert(END, "Select a review window size and click 'Analyze'")
        self.additional_info_text.config(state=DISABLED)
        
        # Clear current results
        self.current_results = None        
    def on_restaurant_selected(self,event=None):
        """
        Handles restaurant selection
        """
         # Don't process if we're in the middle of loading
        if self.is_loading:
            return
        
        restaurant_name = self.restaurant_var.get()
        if not restaurant_name or self.df is None:
            return
        
        try:
            restaurant_df = self.df[self.df['title'] == restaurant_name]
            self.current_restaurant_df = restaurant_df

            total_reviews = len(restaurant_df)
            
            # Update restaurant info
            self.restaurant_name_label.config(text=f"Restaurant: {restaurant_name}")
            self.review_count_label.config(text=f"Total Reviews: {total_reviews}")
            
            # Update date range if available
            if 'publishedAtDate' in restaurant_df.columns:
                dates = pd.to_datetime(restaurant_df['publishedAtDate'])
                date_range = f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}"
                self.date_range_label.config(text=f"Date range: {date_range}")
            
            #Clear the previous results
            self.clear_results_display()
            
            # Update available window sizes
            available_windows = [w for w in self.review_windows if w <= total_reviews]
            if not available_windows:
                available_windows = [total_reviews] if total_reviews > 0 else [1]
        
            self.window_dropdown['values'] = available_windows
            
            if available_windows:
                # Set to first option
                self.window_var.set(available_windows[0])  # Set to smallest, not largest
                self.window_dropdown.config(state="readonly")
            else:
                self.window_var.set(0)
                self.window_dropdown.config(state=DISABLED)
        
            # Enable analyze button
            self.analyze_btn.config(state=NORMAL)
            
            self.status_label.config(text=f"Selected: {restaurant_name}. Now choose review window size.")
            
        except Exception as e:
            print(f"ERROR in on_restaurant_selected: {e}")

    def on_window_selected(self, event=None):
        """Handle window size selection """
        #previous auto analyzes, but user should have choice to pick review window
        window_size = self.window_var.get()
        if window_size > 0:
            # Just update status, don't analyze
            self.status_label.config(text=f"Window size: {window_size} reviews. Click 'Analyze' to run.")
            
    def analyze_restaurant(self):
        """Analyze restaurant using threading"""
        if not hasattr(self, 'current_restaurant_df') or self.model is None:
            return
        
        restaurant_name = self.restaurant_var.get()
        window_size = self.window_var.get()
        
        if not restaurant_name or window_size == 0:
            return
        
        # Disable controls during analysis
        self.restaurant_dropdown.config(state=DISABLED)
        self.window_dropdown.config(state=DISABLED)
        self.analyze_btn.config(state=DISABLED)
        
        # Show analyzing status
        self.status_label.config(text=f"Analyzing {restaurant_name}...")
    
        def analyze_thread():
            """Background thread for analysis"""
            try:
                # Run analysis in background
                analysis = semester_project.analyze_restaurant_multiple_windows(
                    self.model,
                    self.current_restaurant_df,
                    review_windows=[window_size]
                )
                
                # Update GUI in main thread
                if analysis:
                    self.root.after(0, lambda: self.finish_analysis(analysis, window_size, restaurant_name))
                else:
                    self.root.after(0, lambda: self.analysis_failed(f"No analysis available for {restaurant_name}"))
                    
            except Exception as e:
                self.root.after(0, lambda: self.analysis_failed(str(e)))
    
        # Start analysis thread
        threading.Thread(target=analyze_thread, daemon=True).start()

    def finish_analysis(self, analysis, window_size, restaurant_name):
        """Called when analysis completes successfully"""
        self.current_results = analysis
        self.display_results(analysis, window_size)
        
        # Re-enable controls
        self.restaurant_dropdown.config(state="readonly")
        self.window_dropdown.config(state="readonly")
        self.analyze_btn.config(state=NORMAL, bg=self.colors['neutral'])
        
        # Update status and hide progress bar
        self.status_label.config(text=f" Analysis complete for {restaurant_name}")
    
    def analysis_failed(self, error_msg):
        """Handle analysis failure"""
        messagebox.showerror("Analysis Error", error_msg)
        self.status_label.config(text=f"Analysis failed")
        
        # Re-enable controls
        self.restaurant_dropdown.config(state="readonly")
        self.window_dropdown.config(state="readonly")
        self.analyze_btn.config(state=NORMAL, bg=self.colors['neutral'])

    def update_additional_info(self, window_result, analysis):
        """Update the additional statistics text box"""
        self.additional_info_text.config(state=NORMAL)
        self.additional_info_text.delete(1.0, END)
        
        # Create formatted additional info
        additional_info = f"""
        Analysis Details:
        • Window Size: {window_result['window_size']} most recent reviews
        • Reviews Analyzed: {window_result['review_count']} reviews
        • Significant Discrepancy: {'YES' if window_result['significant_discrepancy'] else 'NO'}

        Sentiment Distribution:
        • Positive Ratio: {window_result['positive_ratio']*100:.1f}%
        • Negative Ratio: {window_result['negative_ratio']*100:.1f}%
        • Recent Stars Average: {window_result['recent_stars_avg']:.1f}/5
        • Historical Stars Average: {analysis['historical_stars_avg']:.1f}/5

        Date Range:
        • First Review: {analysis['first_review_date'].strftime('%Y-%m-%d')}
        • Latest Review: {analysis['latest_review_date'].strftime('%Y-%m-%d')}
        """
        
        self.additional_info_text.insert(END, additional_info)
        
        # Highlight significant discrepancy
        if window_result['significant_discrepancy']:
            # Find the line with "Significant Discrepancy"
            self.additional_info_text.tag_add("warning", "4.0", "4.end")
        
        self.additional_info_text.config(state=DISABLED)     
           
    def display_results(self,analysis, window_size):
        # Find the window result for selected size
        window_result = None
        for window in analysis['windows_analyzed']:
            if window['window_size'] == window_size:
                window_result = window
                break
        
        else:
            return
        
        # Use the values calculated in semester_project
        #total score
        total_score = analysis['totalScore']  # Original 0-5 scale
        
        #recent sentiment
        recent_sentiment = window_result['recent_sentiment_mean']  # Already calculated
        recent_stars = ((recent_sentiment + 1) / 2) * 5  # Convert -1:1 → 0:5 (necessary for readability)
        
        #sentiment gap
        gap_raw = window_result['sentiment_gap']  # -1:1 scale
        gap_in_stars = gap_raw * 2.5  # Convert -1:1 gap → -2.5:+2.5 star gap
        
        # Direction is already determined in semester_project
        direction = window_result['direction']
        
        #DISPLAY HERE
        self.total_score_value.config(text=f"{total_score:.1f}/5 stars")
        self.recent_sentiment_value.config(text=f"{recent_stars:.1f}/5")
        
        gap_text = f"{gap_in_stars:+.2f}"
        self.gap_value.config(text=gap_text)

        
        # Color code based on direction
        if direction == 'improving':
            self.gap_value.config(fg=self.colors['positive'])
            self.gap_icon.config(text='\u25b2', fg=self.colors['positive'])
        elif direction == 'declining':
            self.gap_value.config(fg=self.colors['negative'])
            self.gap_icon.config(text='\u25bc', fg=self.colors['negative'])
        else:
            self.gap_value.config(fg=self.colors['neutral'])
            self.gap_icon.config(text='\u25cf', fg=self.colors['neutral'])
            
        if direction == 'improving':
            self.direction_value.config(text="↑ Improving", fg=self.colors['positive'])
        elif direction == 'declining':
            self.direction_value.config(text="↓ Declining", fg=self.colors['negative'])
        else:
            self.direction_value.config(text="→ Stable", fg=self.colors['neutral'])
        
        self.update_additional_info(window_result, analysis)
        
    def show_error(self, message):
        """Show error message"""
        messagebox.showerror("Error", message)
        self.status_label.config(text="Error occurred")
        
def main():
    root = Tk()
    # Center window on screen
    window_width = 800
    window_height = 650
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    center_x = int(screen_width/2 - window_width/2)
    center_y = int(screen_height/2 - window_height/2)
    root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
    
    # Set minimum window size
    root.minsize(700, 550)
    
    #create and run the GUI
    app = RestaurantSentimentGUI(root)
    root.mainloop()
    

if __name__ == "__main__":
    main()