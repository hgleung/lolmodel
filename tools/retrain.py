from model import Model

def main():
    print("Loading model...")
    model = Model()
    
    print("\nRetraining model with current scikit-learn version...")
    model.train_random_forest(force_retrain=True)
    
    print("\nDone! The model has been retrained and saved.")

if __name__ == "__main__":
    main()
