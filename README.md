# F1 Race Winner Prediction Model

## Overview

This project aims to predict the probability of each driver winning a Formula 1 race based on historical data and the starting grid for the upcoming race. It utilizes qualifying and race result data from the 2020 season onwards to train a machine learning model.

The model considers factors such as:
*   Starting grid position
*   Historical performance associated with the driver (via ID)
*   Historical performance associated with the constructor (team) (via ID)
*   The specific circuit (via ID)
*   **Recent driver performance:** Rolling averages (last 5 races) of points scored, finishing position, and grid position.
*   **Driver's championship standing:** Cumulative points scored in the season *before* the race being predicted.

## Features

*   Loads historical F1 qualifying and race data from CSV files.
*   Preprocesses data, including handling missing values, cleaning names, and encoding categorical features.
*   Engineers features like rolling performance metrics and championship standings based on historical context.
*   Trains a `RandomForestClassifier` model using `scikit-learn`.
*   **Saves the trained model** using `joblib` for efficient reuse.
*   **Loads the saved model** for making predictions without retraining each time.
*   Provides helper functions to prepare future race grid data, mapping current driver/team names to historical IDs and handling new entities.
*   Implements a prediction workflow that correctly calculates time-dependent features (rolling averages, standings) based on the history *up to* the race being predicted.
*   Provides a wrapper function to orchestrate the prediction process for a future race.
*   **Saves prediction probabilities** for each race to individual CSV files.
*   Displays prediction results in a clear, formatted Markdown table within a notebook environment.
*   Supports retraining the model as new race data becomes available.

## Data Requirements

The model requires two main CSV files:

1.  **`qualifying.csv`**: (or your updated file): Contains qualifying results, including driver/constructor names and IDs. Must be kept up-to-date if new drivers/teams appear who don't race (unlikely but possible).
    *   Expected columns: `season`, `round`, `date`, `raceName`, `circuitId`, `driverId`, `driverFullName`, `constructorId`, `constructorName`, `qualifyPosition`, `q1`, `q2`, `q3`
2.  **`races.csv`** (or your updated file): Contains race results, including grid positions, final positions, and points. **This file MUST be updated with new race results before retraining.**
    *   Expected columns: `season`, `round`, `date`, `raceName`, `circuitId`, `driverId`, `driverFullName`, `constructorId`, `constructorName`, `grid`, `position`, `points`, `tavg`, `tmin`, `tmax`, `prcp`, `wspd` (weather columns are loaded but not currently used as features).

*   **Important:** Place these CSV files in the same directory as the Jupyter Notebook (`.ipynb`) file.
*   The code expects specific column names as listed above.
*   Date columns should be parseable by pandas (`YYYY-MM-DD` or similar format).

## Installation / Setup

1.  **Python:** Ensure you have Python 3.8 or later installed.
2.  **Libraries:** Install the required libraries using pip:
    ```bash
    pip install pandas numpy scikit-learn ipython joblib
    ```
    *(Note: `joblib` is often included with scikit-learn)*
3.  **Data:** Download or place the `qualifying.csv` and your most up-to-date `races.csv` file in the project directory.
4.  **Prediction Folder:** The script will automatically create a subfolder named `predictions` (or as specified in the `predict_and_display_results` call) to save the CSV outputs.

## Workflow

The project is structured as a Jupyter Notebook (`.ipynb`) file. Run the cells sequentially.

**A. Initial Training / Retraining:**

*   **Run Cell 1 (Imports):** Imports necessary libraries.
*   **Run Cell 2 (Data Loading):** Loads the *latest* `qualifying.csv` and `races.csv` files. **Update the `races.csv` filename in this cell if needed.** Performs basic cleaning.
*   **Run Cell 3 (Feature Engineering Function & Mapping Creation):** Defines `calculate_features` and runs it on the loaded data. Creates essential Name <-> ID mappings.
*   **Run Cell 4 (Model Definition and Preprocessing Setup):** Defines the features (including rolling/standings) and sets up the `scikit-learn` preprocessing pipeline.
*   **Run Cell 5 (Model Training):** Trains the `RandomForestClassifier` model on the prepared data. **Crucially, this cell now saves the trained `model_pipeline` to `f1_winner_predictor_model.joblib`**. Wait for completion.

**B. Making Predictions for a Future Race:**

*   **Run Cells 6, 7, 8 (Function Definitions):** Ensure the helper functions (`prepare_grid_for_prediction`, `predict_race_winner_probabilities`, `predict_and_display_results`) are defined in the notebook's memory.
*   **Run "Define Grid" Cell (e.g., Cell 9, 11):** Create a new cell or modify an existing one to define the `raw_grid_list` (using current driver/team names), `circuit_id`, `future_season`, `future_round`, and `race_description` for the race you want to predict.
*   **Run "Execute Prediction" Cell (e.g., Cell 10, 12):** Create a new cell or modify an existing one. This cell should:
    1.  Load the saved model using `joblib.load("f1_winner_predictor_model.joblib")`.
    2.  Call the `predict_and_display_results` wrapper function, passing the loaded model, the grid definition variables, the base dataframes (`races_df`, `qualifying_df`), the mapping dictionaries, the team rebrand map, and the list of model features.
    *   This function will handle the complex workflow: prepare input -> combine -> recalculate features -> isolate -> predict -> display -> save CSV.

## Model Details

*   **Model Type:** `sklearn.ensemble.RandomForestClassifier`
*   **Target Variable:** `is_winner` (Binary: 1 if `position` = 1, else 0)
*   **Key Features Used:**
    *   `grid`: Starting grid position.
    *   `circuitId`: Categorical ID for the race track.
    *   `driverId`: Categorical ID for the driver.
    *   `constructorId`: Categorical ID for the constructor.
    *   `avg_points_last_5`: Driver's rolling average points (previous 5 races).
    *   `avg_position_last_5`: Driver's rolling average finish position (previous 5 races).
    *   `avg_grid_last_5`: Driver's rolling average grid position (previous 5 races).
    *   `points_standings_prev_race`: Driver's cumulative points in the season before this race.
*   **Preprocessing:**
    *   Categorical features (`circuitId`, `driverId`, `constructorId`) are imputed (most frequent) and then One-Hot Encoded. Unknown categories encountered during prediction are ignored (handled by `handle_unknown='ignore'`).
    *   Numerical features are imputed using the median value.
*   **Prediction Workflow:** For future races, features are recalculated by temporarily appending the future race grid to the historical data to ensure time-dependent features reflect the state *before* the predicted race.

## Limitations

*   **Model Simplicity:** While incorporating rolling metrics, the RandomForest might not capture all complex temporal patterns or interactions.
*   **Feature Scope:** Still lacks weather, tyre strategy, detailed circuit data, qualifying time gaps, incident prediction, reliability modeling.
*   **New Entities:** Predictions for drivers with very limited history (rookies) are less reliable as the model has little specific data to learn from. Placeholder IDs are used.
*   **Cold Start:** Rolling features will be less informative at the very start of a new season (few or no previous races in that season).
*   **Data Sensitivity:** Accuracy depends on the quality/completeness of historical data. The impact of a small amount of new data (e.g., 1-2 races) might be limited when retraining on a large history.
*   **No Formal Evaluation:** Rigorous model evaluation (cross-validation, hold-out sets) is not implemented in this predictive setup but would be crucial for assessing true performance.

## Future Improvements

*   Incorporate weather forecasts as features.
*   Add more sophisticated time-series features or use models designed for sequences (e.g., LSTMs, though likely overkill here).
*   Experiment with Gradient Boosting models (XGBoost, LightGBM).
*   Implement proper model evaluation and hyperparameter tuning.
*   Develop better strategies for handling new drivers (e.g., using average rookie stats, embeddings).
*   Weight recent data more heavily in training or feature calculation.
*   Build a simple interface (e.g., Streamlit) for easier interaction.

## Attribution

Maintained with ❤️ by Adithya D M