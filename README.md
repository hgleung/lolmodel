# League Model Data Collection

A machine learning-based system for predicting League of Legends player performance in professional matches. The model combines historical player statistics, team performance data, and recent match history to generate predictions for kills, assists, deaths, and fantasy points.

Key features:
- Player performance prediction using Random Forest regression
- Automated data collection from multiple sources
- Team-vs-team match outcome analysis
- Win probability adjustments
- Fantasy point calculations

## Usage

Run scripts using the provided `run.py` wrapper to ensure correct Python path:
```bash
python run.py tools/scripts.py  # Run general utilities
python run.py tools/test_match_history.py  # Test match history scraping
```

## Dependencies

- Python >= 3.10
- pandas >= 2.0.0
- numpy >= 2.2.0
- scikit-learn >= 1.6.0
- requests >= 2.31.0
- beautifulsoup4 >= 4.9.2
- lxml >= 4.9.2

## Data Collection History

### Player Statistics (player_stats.csv)
- Last scraped: 2025-03-01 12:35:55

### Team Statistics (team_stats.csv)
- Last scraped: 2025-03-01 12:35:57

## Project Structure

- `model.py`: Main model implementation
- `scrapers/`: Web scraping utilities
- `tools/`: Helper scripts and utilities
- `data/`: CSV files and trained models

## To Do:
- [x] Scrape team data (also gol)
- [x] IndexMatch
- [x] Scrape LX match data (fandom)
- [x] Implement RF regression model
- [x] Alternative stats: assists, deaths, fantasy points
- [ ] Tune model
- [ ] Scrape UD for lines
- [ ] Scrape TP for odds