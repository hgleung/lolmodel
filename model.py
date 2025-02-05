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
        
        # Return the dataframe with only the selected columns
        return df[selected_columns]

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
        result = {}
        
        # Find player in the index (case-insensitive)
        player_info = self.indexmatch[self.indexmatch['PP ID'].str.upper() == player_name.upper()]
        if len(player_info) == 0:
            raise ValueError(f"Player {player_name} not found in index")
        
        # Get player's team and leaguepedia ID
        result['team_name'] = player_info['Team GOL ID'].iloc[0]
        leaguepedia_id = player_info['Leaguepedia ID'].iloc[0]
        
        # Try to get player stats using Leaguepedia ID first (case-insensitive)
        player_stats = self.players[self.players['player'].str.upper() == leaguepedia_id.upper()]
        
        # If not found, try using PP ID as fallback (case-insensitive)
        if len(player_stats) == 0:
            player_stats = self.players[self.players['player'].str.upper() == player_name.upper()]
        
        if len(player_stats) == 0:
            print(f"Warning: Player {player_name} not found in player statistics using either ID, using defaults")
            result['player_stats'] = {
                'games': 0,
                'win_rate': 0.5,
                'kda': 0,
                'avg_kills': 0,
                'avg_deaths': 0,
                'avg_assists': 0,
                'csm': 0,
                'kp%': 0
            }
        else:
            result['player_stats'] = player_stats.iloc[0].to_dict()
        
        # Get team stats
        team_stats = self.teams[self.teams['name'] == result['team_name']]
        if len(team_stats) == 0:
            print(f"Warning: Team {result['team_name']} not found in team statistics, using defaults")
            result['team_stats'] = {
                'games': 0,
                'win_rate': 0.5,
                'k:d': 1.0,
                'game_duration': '30:00',
                'kills_/_game': 0,
                'deaths_/_game': 0
            }
        else:
            result['team_stats'] = team_stats.iloc[0].to_dict()
        
        # Get match history and calculate averages
        try:
            match_history = get_match_history(leaguepedia_id)
            if match_history is not None and len(match_history) > 0:
                # Calculate win averages
                wins = match_history[match_history['result'].str.startswith('W')]
                losses = match_history[match_history['result'].str.startswith('L')]
                
                result['recent_performance'] = {
                    'wins': {
                        'last_3_win_avg': wins.iloc[-3:]['kills'].mean() if len(wins) >= 3 else 0,
                        'last_5_win_avg': wins.iloc[-5:]['kills'].mean() if len(wins) >= 5 else 0,
                        'last_7_win_avg': wins.iloc[-7:]['kills'].mean() if len(wins) >= 7 else 0,
                        'last_9_win_avg': wins.iloc[-9:]['kills'].mean() if len(wins) >= 9 else 0
                    },
                    'losses': {
                        'last_3_loss_avg': losses.iloc[-3:]['kills'].mean() if len(losses) >= 3 else 0,
                        'last_5_loss_avg': losses.iloc[-5:]['kills'].mean() if len(losses) >= 5 else 0,
                        'last_7_loss_avg': losses.iloc[-7:]['kills'].mean() if len(losses) >= 7 else 0,
                        'last_9_loss_avg': losses.iloc[-9:]['kills'].mean() if len(losses) >= 9 else 0
                    },
                    'split_avg': match_history['kills'].mean()
                }
            else:
                print(f"Warning: No match history found for {player_name}, using defaults")
                result['recent_performance'] = {
                    'wins': {
                        'last_3_win_avg': 0,
                        'last_5_win_avg': 0,
                        'last_7_win_avg': 0,
                        'last_9_win_avg': 0
                    },
                    'losses': {
                        'last_3_loss_avg': 0,
                        'last_5_loss_avg': 0,
                        'last_7_loss_avg': 0,
                        'last_9_loss_avg': 0
                    },
                    'split_avg': 0
                }
        except Exception as e:
            print(f"Warning: Error calculating match history averages: {str(e)}, using defaults")
            result['recent_performance'] = {
                'wins': {
                    'last_3_win_avg': 0,
                    'last_5_win_avg': 0,
                    'last_7_win_avg': 0,
                    'last_9_win_avg': 0
                },
                'losses': {
                    'last_3_loss_avg': 0,
                    'last_5_loss_avg': 0,
                    'last_7_loss_avg': 0,
                    'last_9_loss_avg': 0
                },
                'split_avg': 0
            }
        
        return result

    def get_team_name_from_code(self, team_code):
        """Get the full team name from team code"""
        # Case-insensitive team code lookup
        team_data = self.indexmatch[self.indexmatch['Team'].str.upper() == team_code.upper()]
        if team_data.empty:
            raise ValueError(f"Team code {team_code} not found")
        return team_data['Team GOL ID'].iloc[0]

    def _calculate_features(self, player_name, player_team, opponent_team):
        """Calculate all features for prediction"""
        # Player analysis
        player_analysis = self.analyze_player(player_name)
        
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
            'player_split_avg': player_analysis['player_stats']['avg_kills'],
            
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

    def calculate_prediction_features(self, player_name, opponent_team_code):
        """Calculate prediction features for a player against an opponent team"""
        # Get opponent team name from code
        opponent_team = self.get_team_name_from_code(opponent_team_code)
        
        # Get player's current team
        player_data = self.indexmatch[self.indexmatch['PP ID'].str.upper() == player_name.upper()]
        if player_data.empty:
            raise ValueError(f"Player {player_name} not found")
        
        player_team_code = player_data['Team'].iloc[0]
        player_team = self.get_team_name_from_code(player_team_code)
        
        # Calculate features using team names
        features = self._calculate_features(player_name, player_team, opponent_team)
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
    print("Enter input as: <player_name> <opponent_team_code> <win_chance>")
    print("Example: Ruler T1 0.7")
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
                player_name, opponent_team_code, win_chance = user_input.split()
                win_chance = float(win_chance)
                
                # Validate win chance
                if not 0 <= win_chance <= 1:
                    print("Error: Win chance must be between 0 and 1")
                    continue
                
            except ValueError:
                print("Error: Please provide input in the format: player_name/opponent_team_code/win_chance")
                continue
            
            # Calculate prediction
            try:
                features = model.calculate_prediction_features(player_name, opponent_team_code)
                prediction = model.calculate_preliminary_prediction(features, win_chance)
                
                # Get team names for display
                player_team_code = model.indexmatch[model.indexmatch['PP ID'].str.upper() == player_name.upper()]['Team'].iloc[0]
                player_team_name = model.get_team_name_from_code(player_team_code)
                opponent_team_name = model.get_team_name_from_code(opponent_team_code)
                
                print(f"\nPrediction for {player_name} ({player_team_name}) vs {opponent_team_name}")
                print(f"Win chance: {win_chance:.1%}")
                print(f"Predicted kills per game: {prediction:.2f}")
                
            except Exception as e:
                print(f"Error calculating prediction: {str(e)}")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            continue
