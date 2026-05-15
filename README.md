# FIFA MatchLab

FIFA MatchLab is an interactive football analytics platform that predicts international match outcomes using historical FIFA match data. Users can compare teams, view win/draw/loss probabilities, explore performance trends, and simulate World Cup-style matchups.




## Live-Demo

You can view my live deployment through vercel: https://fifa-matchlab.vercel.app/





## Features

- **Match Outcome Predictor**  
  Select two international teams and generate win, draw, and loss probabilities using a machine learning model trained on historical football match data.

- **Interactive Team Selection**  
  Choose from national teams in the dataset using dropdown selectors with country flag support.

- **Prediction Probability Chart**  
  Visualizes the predicted probability of each outcome using an interactive bar chart.

- **Model Explanation Panel**  
  Provides a readable explanation of why the model predicted a specific outcome, based on factors such as recent form, win rate, attack, defence, and goal difference.

- **Upset Alert System**  
  Highlights whether a matchup is stable, uncertain, volatile, or a potential upset based on confidence and team strength indicators.

- **Team Comparison Dashboard**  
  Compares two selected teams across key statistics including win rate, average goals scored, average goals conceded, recent form, and match experience.

- **Radar Comparison Chart**  
  Displays a normalized radar chart comparing both teams across attacking strength, defensive strength, recent form, win rate, and goal difference.

- **Head-to-Head History**  
  Shows historical meetings between the selected teams, including past scores, tournaments, winners, and overall head-to-head record.

- **Team Profile Cards**  
  Displays detailed team statistics such as total matches played, goals scored, goals conceded, average goals, and recent form.

- **Knockout Tournament Simulator**  
  Allows users to simulate a knockout-style tournament using 2, 4, 8, or 16 teams, with each round predicted by the model.

- **Top Teams Rankings**  
  Ranks teams by overall strength, attack, defence, recent form, win rate, and goal difference.

- **Model Performance Section**  
  Displays model accuracy, training/testing data size, classification report, and the most influential features used by the model.

- **Country-Only Dataset Filtering**  
  Filters the dataset to focus on international country/national teams rather than regional or non-country teams.




## Tech Stack

### Frontend
- **Next.js** - React framework used to build the web application
- **React** - Component-based UI development
- **TypeScript** - Type-safe frontend development
- **Tailwind CSS** - Styling and responsive dashboard design
- **Recharts** - Interactive charts and visualizations

### Machine Learning / Data Processing
- **Python** - Core language for data processing and model development
- **Pandas** - Data cleaning, transformation, and feature engineering
- **NumPy** - Numerical operations
- **Scikit-learn** - Machine learning pipeline, preprocessing, and Logistic Regression model
- **Joblib** - Saving and loading trained model artifacts
- **PyCountry** - Filtering and mapping country/national team data

### Model
- **Multiclass Logistic Regression** - Predicts home win, draw, or away win
- **ColumnTransformer** - Handles separate preprocessing for categorical and numeric features
- **OneHotEncoder** - Encodes team names and match types
- **StandardScaler** - Normalizes numerical features

### Data

 - **Kaggle** - used to get data on all international football teams/fixtures since 1872

### Deployment
- **Vercel** - Hosts the live Next.js web app
- **GitHub** - Version control and project repository





## Future Plans

I don't have any huge plans, however I might implement somthing to do with the 2026 world cup with the exact fixtures and simulate the entire world cup. This would be a great test of my model to see how much of it actually translates.



## Project Structure

```txt
fifa-matchlab/
│
├── data/
│   ├── raw/
│   │   └── results.csv                         Historical international football match dataset
│   │
│   └── processed/
│       ├── matches_clean.csv                   Cleaned match results with outcome labels
│       ├── team_stats.csv                      Aggregated team statistics
│       ├── teams.json                          List of teams from the cleaned dataset
│       ├── app_teams.json                      Country teams used in the web app
│       ├── latest_team_features.json           Latest engineered features for each team
│       └── model_predictions.json              Pre-generated match outcome predictions
│
├── model/
│   ├── prepare_data.py                         Cleans raw data and filters country teams
│   ├── train_model.py                          Trains the match outcome prediction model
│   ├── export_app_data.py                      Exports JSON data for the frontend
│   │
│   └── artifacts/
│       ├── match_outcome_model.joblib          Saved Scikit-learn model pipeline
│       └── model_metrics.json                  Accuracy, classification report, and model metrics
│
├── notebooks/
│   └── exploration.ipynb                       Initial data exploration and analysis
│
├── web/
│   ├── app/
│   │   ├── page.tsx                            Main FIFA MatchLab dashboard page
│   │   ├── layout.tsx                          App layout and metadata
│   │   └── globals.css                         Global styling and Tailwind CSS setup
│   │
│   ├── public/
│   │   └── data/
│   │       ├── app_teams.json                  Teams available in the dropdowns
│   │       ├── model_predictions.json          Prediction probabilities for team matchups
│   │       ├── latest_team_features.json       Team profile and comparison data
│   │       ├── head_to_head.json               Historical head-to-head records
│   │       ├── rankings.json                   Team rankings by attack, defence, form, and strength
│   │       ├── model_insights.json             Model accuracy and feature importance data
│   │       ├── model_metrics.json              Model performance metrics
│   │       └── team_metadata.json              Country flags and team metadata
│   │
│   ├── package.json                            Frontend dependencies and scripts
│   ├── tsconfig.json                           TypeScript configuration
│   └── next.config.ts                          Next.js configuration
│
├── requirements.txt                            Python dependencies
├── .gitignore                                  Files and folders ignored by Git
└── README.md                                   Project documentation






