from scrape_match_history import get_match_history, calculate_averages

def test_player_history(player_id):
    print(f"Testing match history for player: {player_id}")
    try:
        # Get match history
        history = get_match_history(player_id)
        
        # Display columns
        print("\nColumns in the dataframe:")
        print(history.columns.tolist())
        
        # Display basic info
        print(f"\nFound {len(history)} matches")
        print("\nFirst few matches:")
        print(history.head())
        
        # Calculate averages
        win_averages = calculate_averages(history, 'Win')
        loss_averages = calculate_averages(history, 'Loss')
        
        print("\nWin Averages:")
        for k, v in win_averages.items():
            print(f"{k}: {v}")
            
        print("\nLoss Averages:")
        for k, v in loss_averages.items():
            print(f"{k}: {v}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_player_history("Aiming")
