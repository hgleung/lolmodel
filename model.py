import pandas as pd
import numpy as np
from scrape_match_history import get_match_history, calculate_averages

class Model:
    def __init__(self):
        self.players = self.load_player_data()
        self.teams = self.load_team_data()
        self.indexmatch = pd.read_csv('data/index_match.csv', keep_default_na=False)

    def load_player_data(self):
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
        model_df = df[selected_columns].copy()  # Create explicit copy
        
        return model_df

    def load_team_data(self):
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
        model_df = df[selected_columns].copy()  # Create explicit copy
        
        # Convert game duration from MM:SS to decimal minutes
        def convert_duration(time_str):
            try:
                minutes, seconds = map(int, time_str.split(':'))
                return minutes + seconds/60
            except:
                return None
        
        model_df['game_duration'] = model_df['game_duration'].apply(convert_duration)
        
        return model_df

    def analyze_player(self, player_name):
        """
        Analyze a player's statistics including team data and recent performance
        
        Args:
            player_name (str): Name of the player to analyze
            
        Returns:
            dict: Dictionary containing player and team statistics
        """
        
        # Find player in the index
        player_info = self.indexmatch[self.indexmatch['PP ID'] == player_name]
        if len(player_info) == 0:
            return {"error": f"Player {player_name} not found in index"}
        
        # Get player's team and leaguepedia ID
        team_name = player_info['Team GOL ID'].iloc[0]
        leaguepedia_id = player_info['Leaguepedia ID'].iloc[0]
        
        # Get player stats
        player_stats = self.players[self.players['player'] == player_name]
        if len(player_stats) == 0:
            return {"error": f"Player {player_name} not found in player statistics"}
        
        # Get team stats
        team_stats = self.teams[self.teams['name'] == team_name]
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

    def calculate_prediction_features(self, player_name, opponent_team):
        """Calculate all features for prediction"""
        # Player analysis
        player_analysis = self.analyze_player(player_name)
        
        player_team = player_analysis['team_name']
        
        return {
            # Player features
            'player_5_game_win_avg': player_analysis['recent_performance']['wins']['last_5_win_avg'],
            'player_5_game_loss_avg': player_analysis['recent_performance']['losses']['last_5_loss_avg'],
            'player_split_avg': player_analysis['player_stats']['average_kills'],
            
            # Team context
            'team_kills_avg': self.teams.loc[self.teams['name'] == player_team, 'kills_/_game'].values[0],
            'team_avg_game_time': self.teams.loc[self.teams['name'] == player_team, 'game_duration'].values[0],
            
            # Opposition context
            'opp_deaths_avg': self.teams.loc[self.teams['name'] == opponent_team, 'deaths_/_game'].values[0],
            'opp_avg_game_time': self.teams.loc[self.teams['name'] == opponent_team, 'game_duration'].values[0]
        }

    def predict_series_kills(self, features, n_games, predicted_wins):
        """Predict total kills using weighted features"""
        # Calculate game time factor
        avg_game_time = (features['team_avg_game_time'] + features['opp_avg_game_time']) / 2
        
        # Calculate components
        win_component = (features['player_5_game_win_avg'] * 0.7 +
                        features['team_kills_avg'] * 0.2 +
                        features['opp_deaths_avg'] * 0.1) * predicted_wins
        
        loss_component = (features['player_5_game_loss_avg'] * 0.5 +
                        features['opp_deaths_avg'] * 0.3) * (n_games - predicted_wins)
        
        # Duration adjustment (normalized to 32.23 minutes, the average game duration globally)
        time_factor = avg_game_time / 32.23
        
        # Calculate total kills
        return (win_component + loss_component) * time_factor

    def calculate_global_avg_duration(self):
        """Calculate the weighted average game duration across all teams"""
        
        # Calculate weighted average using number of games as weights
        total_weighted_duration = (self.teams['game_duration'] * self.teams['games']).sum()
        total_games = self.teams['games'].sum()
        
        avg_duration = total_weighted_duration / total_games if total_games > 0 else 0
        return avg_duration

if __name__ == "__main__":
    # Load the data
    model = Model()

    features = model.calculate_prediction_features("Aiming", "OK BRION")
    predict = model.predict_series_kills(features, 2, 1.4)
    print(predict)

    # Calculate and print global average game duration
    avg_duration = model.calculate_global_avg_duration()
    print(f"Global weighted average game duration: {avg_duration:.2f} minutes")

    # # Example usage
    # player_name = input("Enter player name: ")
    # results = analyze_player(player_name)
    
    # if "error" in results:
    #     print(f"Error: {results['error']}")
    # else:
    #     print("\nPlayer Analysis:")
    #     print(f"Player: {results['player_name']} (Team: {results['team_name']})")
    #     print("\nPlayer Statistics:")
    #     for stat, value in results['player_stats'].items():
    #         print(f"- {stat}: {value}")
    #     print("\nTeam Statistics:")
    #     for stat, value in results['team_stats'].items():
    #         print(f"- {stat}: {value}")
    #     print("\nRecent Performance:")
    #     print("Wins:")
    #     for stat, value in results['recent_performance']['wins'].items():
    #         print(f"- {stat}: {value}")
    #     print("Losses:")
    #     for stat, value in results['recent_performance']['losses'].items():
    #         print(f"- {stat}: {value}")
