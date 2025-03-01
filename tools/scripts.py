import pandas as pd
from model import Model
from scrapers.scrape_match_history import get_match_history
import os
from tools.utils import update_stats

def debug_player(player_name):
    print(f"\nDebugging player: {player_name}")
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # Load index_match.csv
    print("\n1. Checking index_match.csv...")
    index_df = pd.read_csv(os.path.join(data_dir, 'index_match.csv'), keep_default_na=False)
    
    # Check if player exists in index (case-insensitive)
    player_data = index_df[index_df['PP ID'].str.upper() == player_name.upper()]
    print(f"Found {len(player_data)} entries for {player_name} in index_match.csv")
    if not player_data.empty:
        print("Player data:")
        print(player_data.to_string())
        
        # Get Leaguepedia ID
        leaguepedia_id = player_data['Leaguepedia ID'].iloc[0]
        print(f"\n2. Found Leaguepedia ID: {leaguepedia_id}")
        
        # Try to get match history
        print("\n3. Attempting to fetch match history...")
        match_history = get_match_history(leaguepedia_id)
        if match_history is not None:
            print(f"Found {len(match_history)} matches")
            print("\nFirst few matches:")
            print(match_history.head().to_string())
        else:
            print("No match history found")
            
        # Try team lookup
        team_code = player_data['Team'].iloc[0]
        print(f"\n4. Player's team code: {team_code}")
        
        # Initialize model and try team name lookup
        model = Model()
        try:
            team_name = model.get_team_name_from_code(team_code)
            print(f"Team name found: {team_name}")
        except Exception as e:
            print(f"Error looking up team name: {str(e)}")
            
        # Try to calculate features
        print("\n5. Attempting to calculate features...")
        try:
            features = model.calculate_prediction_features(player_name, "T1")  # Using T1 as example opponent
            print("\nFeatures calculated successfully:")
            for key, value in features.items():
                print(f"{key}: {value}")
            
            # Test predictions for different game counts
            print("\n6. Testing predictions for different game counts...")
            win_chances = [0.25, 0.5, 0.75]
            game_counts = [1, 2, 3]
            
            for win_chance in win_chances:
                print(f"\nPredictions with {win_chance*100}% win chance:")
                for num_games in game_counts:
                    pred = model.prediction(features, win_chance, num_games)
                    print(f"For {num_games} game(s):")
                    print(f"  Kills: {pred['kills']:.2f}")
                    print(f"  Assists: {pred['assists']:.2f}")
                    print(f"  Deaths: {pred['deaths']:.2f}")
                    # print(f"  CS: {pred['cs']:.2f}")
                    print(f"  Fantasy Score: {pred['fantasy_score']:.2f}")
            
        except Exception as e:
            print(f"Error calculating features: {str(e)}")
            
    else:
        print(f"Player {player_name} not found in index_match.csv")
        # print("\nAvailable players:")
        # print(sorted(index_df['PP ID'].unique()))

if __name__ == "__main__":
    # Add argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='League Model Scripts')
    subparsers = parser.add_subparsers(dest='action', help='Action to perform')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Debug a player')
    debug_parser.add_argument('player_name', help='Name of the player to debug')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update stats')
    
    args = parser.parse_args()
    
    if args.action == 'update':
        update_stats()
    elif args.action == 'debug':
        debug_player(args.player_name)
    else:
        parser.print_help()