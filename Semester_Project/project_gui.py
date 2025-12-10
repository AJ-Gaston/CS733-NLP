from tkinter import *
import semester_project

class RestaurantSentimentGUI:
    """
    Class that creates the GUI for restaurant sentiment
    """
    def __init__(self, root):
        self.root.title("Restaurant Sentiment Analyzer")
        self.root.geometry("800x600")

def main():
    root = Tk()
    app = RestaurantSentimentGUI(root)
    root.mainloop()
    

if __name__ == "__main__":
    main()