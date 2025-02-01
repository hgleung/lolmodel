import pandas as pd
import numpy as np
from scrape_match_history import get_match_history, calculate_averages
from sklearn.linear_model import LinearRegression

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
        
        features = {
            # Player features
            'player_3_game_win_avg': player_analysis['recent_performance']['wins']['last_3_win_avg'],
            'player_3_game_loss_avg': player_analysis['recent_performance']['losses']['last_3_loss_avg'],
            'player_5_game_win_avg': player_analysis['recent_performance']['wins']['last_5_win_avg'],
            'player_5_game_loss_avg': player_analysis['recent_performance']['losses']['last_5_loss_avg'],
            'player_7_game_win_avg': player_analysis['recent_performance']['wins']['last_7_win_avg'],
            'player_7_game_loss_avg': player_analysis['recent_performance']['losses']['last_7_loss_avg'],
            'player_9_game_win_avg': player_analysis['recent_performance']['wins']['last_9_win_avg'],
            'player_9_game_loss_avg': player_analysis['recent_performance']['losses']['last_9_loss_avg'],
            'player_split_avg': player_analysis['player_stats']['average_kills'],
            
            # Team context
            'team_kills_avg': self.teams.loc[self.teams['name'] == player_team, 'kills_/_game'].values[0],
            'team_avg_game_time': self.teams.loc[self.teams['name'] == player_team, 'game_duration'].values[0],
            
            # Opposition context
            'opp_deaths_avg': self.teams.loc[self.teams['name'] == opponent_team, 'deaths_/_game'].values[0],
            'opp_avg_game_time': self.teams.loc[self.teams['name'] == opponent_team, 'game_duration'].values[0]
        }

        features['team_kills_per_min'] = features['team_kills_avg'] / features['team_avg_game_time']
        features['opp_deaths_per_min'] = features['opp_deaths_avg'] / features['opp_avg_game_time']
        features['k_multi'] = features['opp_deaths_per_min'] / features['team_kills_per_min']

        return features

    def predict_series_kills(self, features, n_games, predicted_wins):
        """Predict total kills using a regression model"""
        # Features to use in prediction
        feature_cols = [
            'player_3_game_win_avg',
            'player_3_game_loss_avg',
            'player_5_game_win_avg',
            'player_5_game_loss_avg',
            'player_7_game_win_avg',
            'player_7_game_loss_avg',
            'player_9_game_win_avg',
            'player_9_game_loss_avg',
            'player_split_avg',
            'team_kills_avg',
            'team_avg_game_time',
            'opp_deaths_avg',
            'opp_avg_game_time',
            'team_kills_per_min',
            'opp_deaths_per_min',
            'k_multi'
        ]
        
        # Print available features
        print("\nAvailable features:")
        for key, value in features.items():
            print(f"{key}: {value}")
        
        # Create feature array
        try:
            X = np.array([features[col] for col in feature_cols]).reshape(1, -1)
            print("\nFeature array shape:", X.shape)
            print("Feature values:", X[0])
        except KeyError as e:
            print(f"Missing feature: {e}")
            return 0
        
        # Manual scaling based on typical ranges
        scaling_ranges = {
            'player_3_game_win_avg': (0, 12),    # Kills typically 0-10
            'player_3_game_loss_avg': (0, 12),
            'player_5_game_win_avg': (0, 12),
            'player_5_game_loss_avg': (0, 12),
            'player_7_game_win_avg': (0, 12),
            'player_7_game_loss_avg': (0, 12),
            'player_9_game_win_avg': (0, 12),
            'player_9_game_loss_avg': (0, 12),
            'player_split_avg': (0, 12),
            'team_kills_avg': (5, 25),           # Team kills typically 5-25
            'team_avg_game_time': (25, 45),      # Game time 25-45 minutes
            'opp_deaths_avg': (5, 25),
            'opp_avg_game_time': (25, 45),
            'team_kills_per_min': (0, 2),        # Kills per minute 0-2
            'opp_deaths_per_min': (0, 2),
            'k_multi': (0.8, 1.2)                # Multiplier typically 0-2
        }
        
        # Scale features manually
        X_scaled = np.zeros_like(X, dtype=float)
        for i, col in enumerate(feature_cols):
            min_val, max_val = scaling_ranges[col]
            X_scaled[0, i] = (X[0, i] - min_val) / (max_val - min_val)
        
        print("\nScaled features:", X_scaled[0])
        
        # Define coefficients
        coefficients = np.array([
            0.15,  # player_3_game_win_avg
            0.10,  # player_3_game_loss_avg
            0.12,  # player_5_game_win_avg
            0.08,  # player_5_game_loss_avg
            0.10,  # player_7_game_win_avg
            0.06,  # player_7_game_loss_avg
            0.08,  # player_9_game_win_avg
            0.04,  # player_9_game_loss_avg
            0.15,  # player_split_avg
            0.20,  # team_kills_avg
            -0.05, # team_avg_game_time
            0.15,  # opp_deaths_avg
            -0.05, # opp_avg_game_time
            0.25,  # team_kills_per_min
            0.20,  # opp_deaths_per_min
            0.15   # k_multi
        ])
        
        # Calculate base prediction
        base_prediction = np.dot(X_scaled, coefficients) * 15  # Scale up to realistic kill numbers
        print("\nBase prediction:", base_prediction)
        
        # Adjust for number of games and predicted wins
        win_ratio = predicted_wins / n_games
        print(f"\nWin ratio: {win_ratio}")
        
        # Final prediction
        prediction = base_prediction * n_games * (1 + (win_ratio - 0.5))
        print(f"\nFinal prediction: {prediction}")
        
        # Ensure prediction is positive and reasonable
        return max(0, prediction[0])

if __name__ == "__main__":
    # Load the data
    model = Model()

    features = model.calculate_prediction_features("Aiming", "OK BRION")
    predict = model.predict_series_kills(features, 2, 1.4)
    print(predict)

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
