import pandas as pd
from model import Model
from scrape_match_history import get_match_history

def debug_player(player_name):
    print(f"\nDebugging player: {player_name}")
    
    # Load index_match.csv
    print("\n1. Checking index_match.csv...")
    index_df = pd.read_csv('data/index_match.csv', keep_default_na=False)
    
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
        except Exception as e:
            print(f"Error calculating features: {str(e)}")
            
    else:
        print(f"Player {player_name} not found in index_match.csv")
        # print("\nAvailable players:")
        # print(sorted(index_df['PP ID'].unique()))

def update_stats():
    """Update both player and team statistics"""
    print("\nUpdating player and team statistics...")
    
    # Run both scraping scripts
    import subprocess
    
    print("\nScraping player stats...")
    subprocess.run(['python3', 'scrape_players.py'], check=True)
    
    print("\nScraping team stats...")
    subprocess.run(['python3', 'scrape_teams.py'], check=True)
    
    print("\nStats update complete!")

if __name__ == "__main__":
    # Add argument parsing
    import argparse
    parser = argparse.ArgumentParser(description='League Model Scripts')
    parser.add_argument('action', choices=['debug', 'update'], help='Action to perform: debug (player analysis) or update (stats)')
    parser.add_argument('--player', help='Player name for debug action')
    
    args = parser.parse_args()
    
    if args.action == 'debug':
        debug_player(args.player)
    elif args.action == 'update':
        update_stats()