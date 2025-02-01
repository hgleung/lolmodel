import pandas as pd
import numpy as np
from scrape_match_history import get_match_history, calculate_averages

def load_player_data():
    # Read the CSV file
    df = pd.read_csv('data/player_stats.csv')
    
    # Select the specified columns
    # Note: Column names are lowercase and use underscores due to our scraping script
    selected_columns = [
        'player',
        'games',
        'win_rate',
        'kda',
        'avg_kills',
        'avg_deaths',
        'avg_assists',
        'csm',  # This is CSPM in the original data
        'kp%'   # This is KP% in the original data
    ]
    
    # Create a new dataframe with only the selected columns
    model_df = df[selected_columns]
    
    return model_df

def load_team_data():
    # Read the CSV file
    df = pd.read_csv('data/team_stats.csv', keep_default_na=False, na_values=["-", "missing"])
    
    # Select the specified columns
    selected_columns = [
        'name',
        'region',
        'games',
        'win_rate',
        'k:d',
        'game_duration',
        'kills_/_game',
        'deaths_/_game',
    ]
        
    # Create a new dataframe with only the selected columns
    model_df = df[selected_columns]
    
    return model_df

def analyze_player(player_name):
    """
    Analyze a player's statistics including team data and recent performance
    
    Args:
        player_name (str): Name of the player to analyze
        
    Returns:
        dict: Dictionary containing player and team statistics
    """
    # Load all necessary data
    players_df = load_player_data()
    teams_df = load_team_data()
    index_df = pd.read_csv('data/index_match.csv', keep_default_na=False)
    
    # Find player in the index
    player_info = index_df[index_df['player_name'] == player_name]
    if len(player_info) == 0:
        return {"error": f"Player {player_name} not found in index"}
    
    # Get player's team and leaguepedia ID
    team_name = player_info['team_name'].iloc[0]
    leaguepedia_id = player_info['leaguepedia_id'].iloc[0]
    
    # Get player stats
    player_stats = players_df[players_df['player'] == player_name]
    if len(player_stats) == 0:
        return {"error": f"Player {player_name} not found in player statistics"}
    
    # Get team stats
    team_stats = teams_df[teams_df['name'] == team_name]
    if len(team_stats) == 0:
        return {"error": f"Team {team_name} not found in team statistics"}
    
    # Get match history and calculate averages
    try:
        match_history = get_match_history(leaguepedia_id)
        win_averages = calculate_averages(match_history, 'Win')
        loss_averages = calculate_averages(match_history, 'Loss')
    except Exception as e:
        win_averages = {"error": str(e)}
        loss_averages = {"error": str(e)}
    
    # Compile results
    results = {
        "player_name": player_name,
        "team_name": team_name,
        "player_stats": {
            "average_kills": player_stats['avg_kills'].iloc[0],
            "games_played": player_stats['games'].iloc[0],
            "win_rate": player_stats['win_rate'].iloc[0]
        },
        "team_stats": {
            "kills_per_game": team_stats['kills_/_game'].iloc[0],
            "average_game_duration": team_stats['game_duration'].iloc[0]
        },
        "recent_performance": {
            "wins": win_averages,
            "losses": loss_averages
        }
    }
    
    return results

if __name__ == "__main__":
    # Load the data
    players = load_player_data()
    teams = load_team_data()
    
    # Display basic information about the player dataset
    print("Player Dataset Shape:", players.shape)
    print("\nFirst few rows:")
    print(players.head())
    print("\nDataset Description:")
    print(players.describe())
    
    # Display any missing values
    print("\nMissing Values:")
    print(players.isnull().sum())

    # Display basic information about the team dataset
    print("\nTeam Dataset Shape:", teams.shape)
    print("\nFirst few rows:")
    print(teams.head())
    print("\nDataset Description:")
    print(teams.describe())
    
    # Display any missing values
    print("\nMissing Values:")
    print(teams.isnull().sum())

    # Example usage
    player_name = input("Enter player name: ")
    results = analyze_player(player_name)
    
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print("\nPlayer Analysis:")
        print(f"Player: {results['player_name']} (Team: {results['team_name']})")
        print("\nPlayer Statistics:")
        for stat, value in results['player_stats'].items():
            print(f"- {stat}: {value}")
        print("\nTeam Statistics:")
        for stat, value in results['team_stats'].items():
            print(f"- {stat}: {value}")
        print("\nRecent Performance:")
        print("Wins:")
        for stat, value in results['recent_performance']['wins'].items():
            print(f"- {stat}: {value}")
        print("Losses:")
        for stat, value in results['recent_performance']['losses'].items():
            print(f"- {stat}: {value}")
