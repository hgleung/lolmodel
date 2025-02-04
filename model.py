import pandas as pd
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

    def calculate_preliminary_prediction(self, features, win_chance):
        """
        Calculate preliminary prediction using weighted averages of player performance
        
        Args:
            features (dict): Dictionary containing player and team statistics
            win_chance (float): Probability of winning (between 0 and 1)
            
        Returns:
            float: Predicted kills per game
        """
        # Basic prediction using split average
        basic_prediction = (
            win_chance * 1.25 * features['player_split_avg'] +
            (1 - win_chance) * 0.75 * features['player_split_avg']
        )
        
        # Weighted win prediction using last N game averages
        win_prediction = (
            features['player_3_game_win_avg'] * 1 +
            features['player_5_game_win_avg'] * 2 +
            features['player_7_game_win_avg'] * 3 +
            features['player_9_game_win_avg'] * 4
        ) / 10
        
        # Adjust win prediction by k_multi
        k_multi_adjustment = (features['k_multi'] - 1) / 5 + 1
        win_prediction *= k_multi_adjustment
        
        # Weighted loss prediction using last N game averages
        loss_prediction = (
            features['player_3_game_loss_avg'] * 1 +
            features['player_5_game_loss_avg'] * 2 +
            features['player_7_game_loss_avg'] * 3 +
            features['player_9_game_loss_avg'] * 4
        ) / 10
        
        # Adjust loss prediction by k_multi
        loss_prediction *= k_multi_adjustment
        
        # Combine predictions based on win chance
        advanced_prediction = win_chance * win_prediction + (1 - win_chance) * loss_prediction
        
        # Average the basic and advanced predictions
        final_prediction = (basic_prediction + advanced_prediction) / 2
        
        return max(0, final_prediction)

if __name__ == "__main__":
    # Load the data
    model = Model()    
    print("\nLeague Prediction Model")
    print("Enter input as: <player_name> <opponent_team> <win_chance>")
    print("Example: Ruler GenG 0.7")
    print("Type 'quit' to exit")
    
    while True:
        try:
            # Get input
            user_input = input("\nEnter prediction parameters: ").strip()
            
            # Check for quit command
            if user_input.lower() == 'quit':
                break
            
            # Parse input
            try:
                player_name, opponent_team, win_chance = user_input.split('/')
                win_chance = float(win_chance)
                
                # Validate win chance
                if not 0 <= win_chance <= 1:
                    print("Error: Win chance must be between 0 and 1")
                    continue
                
            except ValueError:
                print("Error: Please provide input in the format: player_name opponent_team win_chance")
                continue
            
            # Calculate prediction
            try:
                features = model.calculate_prediction_features(player_name, opponent_team)
                prediction = model.calculate_preliminary_prediction(features, win_chance)
                print(f"\nPrediction for {player_name} vs {opponent_team} (Win chance: {win_chance:.1%}):")
                print(f"Predicted kills per game: {prediction:.2f}")
                
            except Exception as e:
                print(f"Error calculating prediction: {str(e)}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            continue
