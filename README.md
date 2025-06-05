
# F1 Race Winner Prediction

## Overview
A machine learning pipeline to predict Formula 1 race winners using a Gradient Boosting Classifier trained on historical race and qualifying data (2020–2025). The project:
- Loads and cleans race and qualifying datasets (data/processed/updated_races.csv, data/processed/updated_qualifying.csv).
- Engineers features such as rolling averages of points, positions, grid positions, and pre-race championship standings.
- Trains a `GradientBoostingClassifier` to estimate win probabilities for each grid entry.
- Provides reusable functions to prepare a future race grid, recalculate features, run predictions, display results in Markdown tables, and save predictions to CSV.

## Repository Structure
```
.
├── data
│   └── processed
│       ├── updated_qualifying.csv
│       └── updated_races.csv
├── joblogs
│   └── f1_winner_predictor_model_gbc.joblib
├── predictions_GBC
│   └── (generated CSVs of future race predictions)
├── notebooks
│   └── f1_winner_prediction.ipynb
├── README.md
└── requirements.txt
```
- **data/processed/**
  - `updated_qualifying.csv` – cleaned qualifying results with `driverFullName`, `constructorName`, and `date`.
  - `updated_races.csv` – cleaned race results with `position`, `points`, and `date`.
- **joblogs/**
  - Saved model pipelines (e.g., `f1_winner_predictor_model_gbc.joblib`).
- **predictions_GBC/**
  - Outputs of future race predictions (CSV files named `<season>_R<round>_<race>_<circuit>_predictions_GBC.csv`).
- **notebooks/**
  - Jupyter Notebook containing all data loading, cleaning, feature engineering, model training, and prediction code.
- **requirements.txt**
  - List of Python dependencies.

## Prerequisites
- Python 3.8+ (tested on 3.9)
- Package manager (pip or conda)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/f1-race-prediction.git
   cd f1-race-prediction
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate       # MacOS/Linux
   venv\Scripts\activate.bat    # Windows
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` should include:
   ```
   pandas
   numpy
   scikit-learn
   joblib
   ```

4. **Data Preparation**  
   Place the following files under `data/processed/`:
   - `updated_qualifying.csv`
   - `updated_races.csv`

   Each CSV must include:
   - A `date` column parseable by `pd.to_datetime`.
   - In `updated_races.csv`: `season`, `round`, `driverId`, `constructorId`, `position`, `points`, `grid`.
   - In `updated_qualifying.csv`: `season`, `round`, `driverId`, `constructorId`, `driverFullName`, `constructorName`, and `date`.

## Data Cleaning & Feature Engineering
- **Driver/Constructor Name Normalization**  
  The notebook defines `clean_driver_name()` to:
  - Remove suffixes (e.g., “ Jr.”).
  - Standardize special characters (e.g., replace `Hülkenberg` → `Hulkenberg`).
  - Map variations (e.g., both lowercase “antonelli” and “Kimi Antonelli” to `Kimi Antonelli`).

- **calculate_features(df_races, df_qualifying)**  
  - Sorts and merges full names if missing.
  - Creates a binary target `is_winner` (1 if `position == 1`, else 0).
  - Replaces `grid == 0` with `21` (indicating PNC start) and fills missing grids with 21.
  - Computes rolling averages (window = 5) for `points`, `position`, and `grid`, shifted by one race to avoid leakage:
    - `avg_points_last_5`, `avg_position_last_5`, `avg_grid_last_5`
  - Calculates cumulative points per season and shifts by one race to get `points_standings_prev_race`.
  - Fills NaNs:  
    - Rolling averages → 0 (for points) or 21 (for position/grid).  
    - `points_standings_prev_race` → 0.

## Model Training
- Located in **Cell 5** of `notebooks/f1_winner_prediction.ipynb`.
- Features used:
  ```
  [
    "grid",
    "circuitId",
    "driverId",
    "constructorId",
    "avg_points_last_5",
    "avg_position_last_5",
    "avg_grid_last_5",
    "points_standings_prev_race",
  ]
  ```
- **Preprocessing**  
  - **Numerical features** (`grid`, `avg_points_last_5`, `avg_position_last_5`, `avg_grid_last_5`, `points_standings_prev_race`) → `SimpleImputer(strategy="median")`.
  - **Categorical features** (`circuitId`, `driverId`, `constructorId`) → `Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])`.
  - Combined via `ColumnTransformer`.
- **Classifier**: `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)`.
- **Full pipeline**:  
  ```python
  model_pipeline = Pipeline(steps=[
      ("preprocessor", preprocessor),
      ("classifier", GradientBoostingClassifier(...)),
  ])
  ```
- After training, the pipeline is saved as:  
  ```
  joblogs/f1_winner_predictor_model_gbc.joblib
  ```

### To Train the Model
1. Open `notebooks/f1_winner_prediction.ipynb`.
2. Run cells sequentially up to **Cell 5** (“Model Training”).
3. Ensure that both `updated_qualifying.csv` and `updated_races.csv` are present in `data/processed/` before running.
4. After training, verify that `joblogs/f1_winner_predictor_model_gbc.joblib` exists.

## Predicting a Future Race

### 1. Prepare a Raw Grid
A “raw grid” is a Python list of dictionaries with keys:
```python
[
    {"driver": <driverFullName>, "team": <constructorName>, "grid": <grid_position>},
    ...
]
```
Example for Albert Park 2025:
```python
albert_park_2025_raw_grid = [
    {"driver": "Lando Norris",  "team": "McLaren Mercedes",           "grid": 1},
    {"driver": "Oscar Piastri", "team": "McLaren Mercedes",           "grid": 2},
    {"driver": "Max Verstappen","team": "Red Bull Racing Honda RBPT", "grid": 3},
    # … (add all 20 entries)
]
```

### 2. Team Rebrand Map
Define any constructor name changes for the current season so that they map back to historical `constructorId`:
```python
TEAM_REBRAND_MAP = {
    "Red Bull Racing Honda RBPT": "red_bull",
    "McLaren Mercedes": "mclaren",
    "Ferrari": "ferrari",
    # … all other current-season names → historical ID strings …
}
```

### 3. Use `predict_and_display_results`
Call this wrapper to:
- Map raw grid (full names) to `driverId`/`constructorId` (using historical maps from training data).
- Combine the future grid with historical `races_df` and recalculate features.
- Run the model to predict win probabilities.
- Display results in a Markdown table.
- Save probabilities to CSV in `predictions_GBC/`.

#### Example
```python
from joblib import load
from notebooks.f1_winner_prediction import (
    predict_and_display_results,
    latest_driver_name_to_id_map,
    latest_constructor_name_to_id_map,
    TEAM_REBRAND_MAP,
    features,
)

# Load historical DataFrames (races_df & qualifying_df) exactly as in Cell 2 of the notebook.
# Load trained model:
model = load("joblogs/f1_winner_predictor_model_gbc.joblib")

# Define raw grid, e.g. albert_park_2025_raw_grid (20 entries)
# Then call:
predict_and_display_results(
    circuit_id="albert_park",
    future_season=2025,
    future_round=1,
    raw_grid_list=albert_park_2025_raw_grid,
    model=model,
    base_races_df=races_df,
    base_qualifying_df=qualifying_df,
    driver_name_to_id_hist_map=latest_driver_name_to_id_map,
    constructor_name_to_id_hist_map=latest_constructor_name_to_id_map,
    team_rebrand_map_current=TEAM_REBRAND_MAP,
    model_features_list=features,
    race_description="2025 Australian Grand Prix",
    save_path="predictions_GBC"
)
```
- **Outputs**  
  - A Markdown‐formatted table showing Driver, Grid, Team, and Probability.
  - CSV file: `predictions_GBC/2025_R01_2025_Australian_Grand_Prix_albert_park_predictions_GBC.csv`.

## Dependencies
List of core packages required (also in `requirements.txt`):
```
pandas>=1.3.0
numpy>=1.19.0
scikit-learn>=1.0.0
joblib>=1.0.0
```
(Optional: `IPython` for display utilities in notebook, but not required for command-line scripts.)

## Directory Creation
- **joblogs/**: Created automatically if not present when saving `*.joblib`.
- **predictions_GBC/**: Created automatically by `predict_and_display_results` if not present.

## Tips & Troubleshooting
- **FileNotFoundError**  
  - Ensure `updated_qualifying.csv` and `updated_races.csv` are placed under `data/processed/`.
- **Model Loading Errors**  
  - If `joblogs/f1_winner_predictor_model_gbc.joblib` is missing, run the training cell first.
- **New Drivers/Teams**  
  - Any unmapped driver or constructor in the raw grid will generate a placeholder ID (e.g., `new_driver_<name>`), but their features will default to zeros. Update `TEAM_REBRAND_MAP` or historical name maps if possible.
- **Empty Probability Issues**  
  - If the model outputs zero‐probabilities for all grid entries, the code automatically assigns equal probability.

## Contributing
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Update or add any tests if applicable.
5. Submit a pull request describing your changes.

## Contact
For questions or issues, please open an issue on GitHub or reach out to the project maintainer.

---
*This README provides all necessary steps to set up, train, and run the F1 race winner prediction pipeline. Follow each section in order to get started quickly.*
