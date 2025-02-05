import pandas as pd
from scrape_match_history import get_match_history, calculate_averages
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import joblib
import os

class Model:
    def __init__(self):
        """Initialize the model by loading necessary data"""
        self.players = self.load_player_data()
        self.teams = self.load_team_data()
        self.indexmatch = pd.read_csv('data/index_match.csv', keep_default_na=False)
        self.rf_model = None
        self.model_path = 'data/rf_model.joblib'
        self.feature_columns = [
            'avg_kills', 'win_rate', 'kda', 'team_kills_avg', 
            'team_kills_per_min', 'player_3_game_win_avg', 
            'player_5_game_win_avg', 'player_3_game_loss_avg', 
            'player_5_game_loss_avg'
        ]
        
        # Try to load pre-trained model
        if os.path.exists(self.model_path):
            try:
                print("Loading pre-trained model...")
                self.rf_model = joblib.load(self.model_path)
                print("Model loaded successfully")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                self.rf_model = None

    def load_player_data(self):
        """Load and preprocess player statistics"""
        selected_columns = [
            'player',
            'games',
            'win_rate',
            'kda',
            'avg_kills',
            'avg_deaths',
            'avg_assists',
            'csm',
            'kp%'   # This is KP% in the original data
        ]
        return pd.read_csv('data/player_stats.csv')[selected_columns]

    def load_team_data(self):
        """Load and preprocess team statistics"""
        return pd.read_csv('data/team_stats.csv')

    def _get_team_stats(self, team_name):
        """Helper to get team stats"""

        team_stats = self.teams[self.teams['name'] == team_name]
        if len(team_stats) == 0:
            # print(f"Warning: Team {team_name} not found in team statistics, using defaults")
            return
        return team_stats.iloc[0]

    def _parse_game_time(self, time_str):
        """Convert MM:SS format to minutes as float"""
        minutes, seconds = map(float, time_str.split(':'))
        return minutes + seconds/60

    def get_team_name_from_code(self, team_code):
        """Get the full team name from team code"""
        # Case-insensitive team code lookup
        team_data = self.indexmatch[self.indexmatch['Team'].str.upper() == team_code.upper()]
        if team_data.empty:
            raise ValueError(f"Team code {team_code} not found")
        return team_data['Team GOL ID'].iloc[0]

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
            # print(f"Warning: Player {player_name} not found in player statistics using either ID")
            return
        else:
            result['player_stats'] = player_stats.iloc[0].to_dict()

        # Get match history and calculate averages
        try:
            match_history = get_match_history(leaguepedia_id)
            if match_history is not None and len(match_history) > 0:
                # Calculate averages for wins and losses
                win_averages = calculate_averages(match_history, 'Win')
                loss_averages = calculate_averages(match_history, 'Loss')

                result['recent_performance'] = {
                    'wins': win_averages,
                    'losses': loss_averages,
                    'split_avg': result['player_stats']['avg_kills']  # Use avg_kills from player_stats
                }
            else:
                # print(f"Warning: No match history found for {player_name}, using defaults")
                return
        except Exception as e:
            # print(f"Warning: Error calculating match history averages: {str(e)}, using defaults")
            return
        
        return result

    def _calculate_features(self, player_name, player_team, opponent_team):
        """Calculate features for prediction"""
        features = {}

        # Get player analysis
        player_analysis = self.analyze_player(player_name)

        # Get win/loss averages
        recent_perf = player_analysis['recent_performance']
        for n in [3, 5, 7, 9]:
            features[f'player_{n}_game_win_avg'] = recent_perf['wins'][f'last_{n}_win_avg']
            features[f'player_{n}_game_loss_avg'] = recent_perf['losses'][f'last_{n}_loss_avg']
        features['player_split_avg'] = recent_perf['split_avg']

        # Get team stats
        team_stats = self._get_team_stats(player_team)
        features['team_kills_avg'] = team_stats['kills_/_game']
        features['team_avg_game_time'] = self._parse_game_time(team_stats['game_duration'])

        # Get opponent stats
        if opponent_team is not None:
            opp_stats = self._get_team_stats(opponent_team)
            features['opp_deaths_avg'] = opp_stats['deaths_/_game']
            features['opp_avg_game_time'] = self._parse_game_time(opp_stats['game_duration'])

        # Calculate derived features
        features['team_kills_per_min'] = features['team_kills_avg'] / features['team_avg_game_time']
        if opponent_team is not None:
            features['opp_deaths_per_min'] = features['opp_deaths_avg'] / features['opp_avg_game_time']
            features['k_multi'] = features['opp_deaths_per_min'] / features['team_kills_per_min'] if features['team_kills_per_min'] > 0 else 1

        return features

    def calculate_prediction_features(self, player_name, opponent_team_code):
        """Calculate all features needed for prediction"""
        # Get team names
        opponent_team = self.get_team_name_from_code(opponent_team_code)

        # Get player's current team
        player_data = self.indexmatch[self.indexmatch['PP ID'].str.upper() == player_name.upper()]
        if player_data.empty:
            raise ValueError(f"Player {player_name} not found")

        player_team = player_data['Team GOL ID'].iloc[0]

        # Calculate features
        features = self._calculate_features(player_name, player_team, opponent_team)
        return features

    def prediction(self, features, win_chance):
        """
        Calculate preliminary prediction using weighted averages of player performance

        Args:
            features (dict): Dictionary of calculated features
            win_chance (float): Probability of winning (0-1)

        Returns:
            float: Predicted kills for the player
        """
        # Basic prediction using split average
        basic_prediction = (
            win_chance * 1.25 * features['player_split_avg'] +
            (1 - win_chance) * 0.75 * features['player_split_avg']
        )

        # Adjust k_multi for predictions
        k_multi_adjustment = (features['k_multi'] - 1) / 5 + 1

        # Calculate weighted win prediction
        win_prediction = (
            features['player_3_game_win_avg'] * 1 +
            features['player_5_game_win_avg'] * 2 +
            features['player_7_game_win_avg'] * 3 +
            features['player_9_game_win_avg'] * 4
        ) / 10
        win_prediction *= k_multi_adjustment

        # Calculate weighted loss prediction
        loss_prediction = (
            features['player_3_game_loss_avg'] * 1 +
            features['player_5_game_loss_avg'] * 2 +
            features['player_7_game_loss_avg'] * 3 +
            features['player_9_game_loss_avg'] * 4
        ) / 10
        loss_prediction *= k_multi_adjustment

        # Combine predictions based on win chance
        advanced_prediction = win_chance * win_prediction + (1 - win_chance) * loss_prediction

        # Average basic and advanced predictions
        final_prediction = (basic_prediction + advanced_prediction) / 2

        return max(0, final_prediction)

    def train_random_forest(self, force_retrain=False):
        """
        Train a random forest model using player data and match history
        
        Args:
            force_retrain (bool): If True, retrain model even if a saved model exists
        """
        # Check if we already have a trained model
        if not force_retrain and self.rf_model is not None:
            print("Using existing trained model")
            return self.rf_model
        else:
            print("Training random forest model...")

        training_data = []
        total_players = len(self.indexmatch)
        processed = 0
        
        # Collect training data from all players
        for _, row in self.indexmatch.iterrows():
            try:
                player_name = row['PP ID']
                team_name = row['Team GOL ID']
                
                # Update progress
                processed += 1
                print(f"\rProcessing {player_name:<20} ({processed}/{total_players} players)", end='', flush=True)
                
                # Get player analysis using existing method
                player_analysis = self.analyze_player(player_name)
                if not player_analysis:
                    continue
                    
                # Calculate features using existing method
                features = self._calculate_features(player_name, team_name, None)
                if not features:
                    continue
                
                # Extract relevant features
                sample = {
                    'avg_kills': float(player_analysis['player_stats'].get('avg_kills')),
                    'win_rate': float(player_analysis['player_stats'].get('win_rate')),
                    'kda': float(player_analysis['player_stats'].get('kda')),
                    'team_kills_avg': float(features.get('team_kills_avg')),
                    'team_kills_per_min': float(features.get('team_kills_per_min')),
                    'player_3_game_win_avg': float(features.get('player_3_game_win_avg')),
                    'player_5_game_win_avg': float(features.get('player_5_game_win_avg')),
                    'player_3_game_loss_avg': float(features.get('player_3_game_loss_avg')),
                    'player_5_game_loss_avg': float(features.get('player_5_game_loss_avg')),
                    'target_kills': float(player_analysis['player_stats'].get('avg_kills'))
                }
                
                # Ensure all required features are present
                if all(col in sample for col in self.feature_columns):
                    training_data.append(sample)
                
            except Exception as e:
                # print(f"\nError processing player {row['PP ID']}: {str(e)}")
                continue
        
        print()  # New line after progress
        
        if not training_data:
            print("Error: No valid training data collected")
            return None
            
        # Convert to DataFrame and ensure feature order
        df = pd.DataFrame(training_data)
        X = df[self.feature_columns].values  # Convert to numpy array without column names
        y = df['target_kills'].values
        
        # Train model
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.rf_model.fit(X, y)
        print(f"Successfully trained random forest model on {len(training_data)} samples")
        
        # Save model to disk
        try:
            print("Saving model to disk...")
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.rf_model, self.model_path)
            print("Model saved successfully")
        except Exception as e:
            print(f"Error saving model: {str(e)}")
        
        return self.rf_model

    def predict_rf(self, player_name, opponent_team_code, win_chance=0.5):
        """Make a prediction using the random forest model"""
        if self.rf_model is None:
            print("Training random forest model first...")
            self.train_random_forest()
            if self.rf_model is None:
                return None
        
        try:
            # Get player analysis
            player_analysis = self.analyze_player(player_name)
            if not player_analysis:
                return None
                
            # Get player's team
            player_data = self.indexmatch[self.indexmatch['PP ID'].str.upper() == player_name.upper()]
            if player_data.empty:
                return None
                
            # Get team names
            try:
                player_team = self.get_team_name_from_code(player_data['Team'].iloc[0])
                opponent_team = self.get_team_name_from_code(opponent_team_code)
            except ValueError as e:
                print(f"Error getting team names: {str(e)}")
                return None
            
            # Calculate features
            features = self._calculate_features(player_name, player_team, opponent_team)
            if not features:
                return None
            
            # Prepare features for prediction using same columns as training
            feature_dict = {
                'avg_kills': float(player_analysis['player_stats'].get('avg_kills')),
                'win_rate': float(player_analysis['player_stats'].get('win_rate')),
                'kda': float(player_analysis['player_stats'].get('kda')),
                'team_kills_avg': float(features.get('team_kills_avg')),
                'team_kills_per_min': float(features.get('team_kills_per_min')),
                'player_3_game_win_avg': float(features.get('player_3_game_win_avg')),
                'player_5_game_win_avg': float(features.get('player_5_game_win_avg')),
                'player_3_game_loss_avg': float(features.get('player_3_game_loss_avg')),
                'player_5_game_loss_avg': float(features.get('player_5_game_loss_avg'))
            }
            
            # Create numpy array with features in correct order
            X = pd.DataFrame([feature_dict])[self.feature_columns].values
            
            # Get predictions from all trees
            predictions = []
            for estimator in self.rf_model.estimators_:
                predictions.append(estimator.predict(X)[0])
            
            # Calculate prediction and confidence
            mean_pred = np.mean(predictions)
            std_pred = np.std(predictions)
            confidence = 1 - (std_pred / mean_pred) if mean_pred > 0 else 0
            
            return {
                'prediction': max(0, mean_pred),  # Ensure non-negative prediction
                'confidence': confidence,
                'range': (max(0, mean_pred - 1.96 * std_pred), mean_pred + 1.96 * std_pred)
            }
            
        except Exception as e:
            print(f"Error making RF prediction: {str(e)}")
            return None

if __name__ == "__main__":
    # Load the data
    model = Model()
    print("\nLeague Prediction Model")
    print("Enter input as: <player_name> <opponent_team_code> <win_chance>")
    print("Example: Ruler HLE 0.7")
    print("Type 'quit' to exit")

    while True:
        try:
            # Get user input
            user_input = input("\nEnter prediction parameters: ").strip()
            if user_input.lower() == 'q':
                break

            player_name, opponent_team_code, win_chance = user_input.split()
            win_chance = float(win_chance)

            # Make predictions
            try:
                print("\nMaking predictions...")
                print(f"Player: {player_name}")
                print(f"Opponent: {opponent_team_code}")
                print(f"Win chance: {win_chance:.1%}")

                # Make baseline prediction
                baseline_pred = model.prediction(model.calculate_prediction_features(player_name, opponent_team_code), win_chance)
                print(f"Baseline prediction: {baseline_pred:.2f} kills")

                # Make RF prediction
                rf_pred = model.predict_rf(player_name, opponent_team_code, win_chance)
                if rf_pred:
                    print(f"RF prediction: {rf_pred['prediction']:.2f} kills (confidence: {rf_pred['confidence']:.2f})")
                    print(f"RF prediction range: ({rf_pred['range'][0]:.2f}, {rf_pred['range'][1]:.2f})")

            except Exception as e:
                print(f"Error calculating prediction: {str(e)}")

        except ValueError as e:
            print(f"Invalid input: {str(e)}")
        except KeyboardInterrupt:
            break
