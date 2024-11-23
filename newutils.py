import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from catboost import CatBoostRegressor, CatBoostClassifier
import matplotlib.pyplot as plt
from glob import glob
import os
import logging
from datetime import datetime
from warnings import filterwarnings
filterwarnings('ignore')
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchPredictor:
    def __init__(self):
        pass

    def get_season(self, month):
        if month in [12, 1, 2]:
            return 0
        elif month in [3, 4, 5]:
            return 1
        elif month in [6, 7, 8]:
            return 2
        else:
            return 3

    def time_of_day(self, hour):
        if 0 <= hour < 6:
            return 0
        elif 6 <= hour < 12:
            return 1
        elif 12 <= hour < 18:
            return 2
        else:
            return 3

    def make_datetime_features(self, df):
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['day_of_week'] = df['date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['season'] = df['month'].apply(self.get_season)
        df['is_month_start_end'] =( (df['date'].dt.is_month_start)| (df['date'].dt.is_month_end)).astype(int)
        
        #временные фичи
        df['time'] = pd.to_datetime(df['time']).dt.time
        df['hour'] = df['time'].apply(lambda x: x.hour)
        df['minute'] = df['time'].apply(lambda x: x.minute) 
        df['time_of_day'] = df['hour'].apply(self.time_of_day)

        df['is_work_time'] = df['hour'].between(9, 18, inclusive='left').astype(int)
        df['decimal_time'] = df['hour'] + df['minute'] / 60.0
        df['is_afternoon'] = (df['hour'] >= 12).astype(int)

        df['days_until_weekend'] = (5 - df['day_of_week']) % 7
        df['days_until_end_of_month'] = df['date'].dt.days_in_month - df['day_of_month']
        
        return df

    def make_team_features(self, df):
        df['avg_scorediff_by_team'] = df.groupby('team2')['scorediff'].transform('mean')
        df['games_played_with_team'] = df.groupby('team2')['team2'].transform('count')
        df['cumulative_scorediff_with_team'] = df.groupby('team2')['scorediff'].cumsum()
        df['win_ratio_with_team'] = df.groupby('team2')['scorediff'].transform(lambda x: (x > 0).sum() / len(x))
        df['avg_scorediff_by_season'] = df.groupby('season')['scorediff'].transform('mean')
        return df

    def make_team_test_features(self, sport_name, df):
        train = pd.read_csv(f'data/{sport_name}.csv')
        user = df['team2'].values.tolist()[0]
        if len(train[train['team2'] == user]) > 0:
            train = train[train.team2 == user]
            df['avg_scorediff_by_team'] = train.groupby('team2')['scorediff'].mean()
            df['games_played_with_team'] = train.groupby('team2')['team2'].count()
            df['cumulative_scorediff_with_team'] = train.groupby('team2')['scorediff'].cumsum().mean()
            df['win_ratio_with_team'] = (train['scorediff'] > 0).sum() / len(train)
            df['avg_scorediff_by_season'] = train.groupby('season')['scorediff'].mean()
        else:
            df['avg_scorediff_by_team'] = 0
            df['games_played_with_team'] = 0
            df['cumulative_scorediff_with_team'] = 0
            df['win_ratio_with_team'] = -1
            df['avg_scorediff_by_season'] = 0
        return df
    

    def train_model(self, sport_name: str):
        df = pd.read_csv(f'{sport_name}.csv')

        # Process datetime and team features
        df = self.make_datetime_features(df)
        df['scorediff'] = df['score1'] - df['score2']
        df = self.make_team_features(df)

        # Select relevant columns
        feature_cols = [
            'year', 'day_of_week', 'is_weekend', 'day_of_month',
            'month', 'quarter', 'day_of_year', 'week_of_year', 'season',
            'is_month_start_end', 'hour', 'minute', 'time_of_day',
            'is_work_time', 'decimal_time', 'is_afternoon',
            'days_until_weekend', 'days_until_end_of_month',
            'avg_scorediff_by_team', 'games_played_with_team',
            'cumulative_scorediff_with_team', 'win_ratio_with_team',
            'avg_scorediff_by_season']
        # Save data
        df.to_csv(f'data/{sport_name}.csv')
        
        X = df[feature_cols]
        y_reg = df['scorediff']
        y_clf = [2 if i > 0 else 1 if i == 0 else 0 for i in df['scorediff'].astype(int)]
        
        # Initialize models
        reg_model = CatBoostRegressor(eval_metric='MAE', verbose=False)
        clf_model = CatBoostClassifier(eval_metric='TotalF1', verbose=False)
        
        # Train regressor
        reg_model.fit(X, y_reg)
        reg_model.save_model(f'models/{sport_name}/{sport_name}_regressor.cbm')
        logger.info(f"Regressor for {sport_name} trained and saved.")
        
        # Train classifier
        clf_model.fit(X, y_clf)
        clf_model.save_model(f'models/{sport_name}/{sport_name}_classifier.cbm')
        logger.info(f"Classifier for {sport_name} trained and saved.")
        return

    def remake_data(self, sport_name: str, new_string_to_df: list):
        try:
            df = pd.read_csv(f'{sport_name}.csv')
        except FileNotFoundError:
            logger.error(f"File for sport '{sport_name}' not found.")
            return
        
        new_string_to_df = new_string_to_df.split(' ')
        new_string_to_df[-1] = new_string_to_df[-1].replace(':', ' ')
        allnew = new_string_to_df[:-1]

        score1, score2 = int(new_string_to_df[-1].split()[0]), int(new_string_to_df[-1].split()[1])
        allnew.extend([score1, score2])
        df.loc[len(df.index)] = allnew
        df.to_csv(f'{sport_name}.csv', index=False)
        logger.info(f"New match data added to {sport_name}.csv.")
        return


    def save_all_plots(self, sport_name: str, plots: bool = False, save: bool = False):
        """
        Generates and saves various plots for the specified sport.

        Parameters:
            sport_name (str): The name of the sport ('football' or 'hockey').
            plots (bool): Whether to display the plots.
            save (bool): Whether to save the plots as PNG files.
        """
        try:
            df = pd.read_csv(f'data/{sport_name}.csv')
        except FileNotFoundError:
            logger.error(f"File for sport '{sport_name}' not found.")
            return
        
        # Calculate average score difference
        average_score = df['scorediff'].mean()
        
        # Plot 1: Score difference vs. Month
        plt.figure(figsize=(12, 6))
        plt.scatter(df['month'], df['scorediff'], label='Разница пропущенных и забитых голов', color='blue', alpha=0.7)
        plt.axhline(y=average_score, color='red', linestyle='--', label=f'Средний результат: {average_score:.2f}')
        plt.title('Результаты игр команды по месяцам')
        plt.xlabel('Месяц')
        plt.ylabel('Разница очков')
        plt.legend()
        plt.grid(alpha=0.5)
        if save:
            plt.savefig(f'plots/{sport_name}/{sport_name}_month_scatter_plot.png')
        if plots:
            plt.show()
        plt.close()
        
        # Plot 2: Score difference vs. Day of Week
        plt.figure(figsize=(12, 6))
        plt.scatter(df['day_of_week'] + 1, df['scorediff'], label='Разница пропущенных и забитых голов', color='blue', alpha=0.7)
        plt.axhline(y=average_score, color='red', linestyle='--', label=f'Средний результат: {average_score:.2f}')
        plt.title('Результаты игр команды по дням недели')
        plt.xlabel('День недели')
        plt.ylabel('Разница очков')
        plt.legend()
        plt.grid(alpha=0.5)
        if save:
            plt.savefig(f'plots/{sport_name}/{sport_name}_day_of_week_scatter_plot.png')
        if plots:
            plt.show()
        plt.close()
        
        # Plot 3: Score difference vs. Time of Day
        plt.figure(figsize=(12, 6))
        print(df['time_of_day'].nunique())
        plt.scatter(df['time_of_day'], df['scorediff'], label='Разница пропущенных и забитых голов', color='blue', alpha=0.7)
        plt.axhline(y=average_score, color='red', linestyle='--', label=f'Средний результат: {average_score:.2f}')
        plt.title('Результаты игр команды по времени дня')
        plt.xlabel('Время дня')
        plt.ylabel('Разница очков')
        plt.legend()
        plt.grid(alpha=0.5)
        if save:
            plt.savefig(f'plots/{sport_name}/{sport_name}_time_of_day_scatter_plot.png')
        if plots:
            plt.show()
        plt.close()
        
        # Plot 4: Average Score Difference by Month
        avg_scorediff_by_month = df.groupby('month')['scorediff'].mean()
        plt.figure(figsize=(10, 6))
        avg_scorediff_by_month.plot(kind='bar')
        plt.title('Средняя разница пропущенных и забитых голов по месяцам')
        plt.xlabel('Месяц')
        plt.ylabel('Разница очков')
        plt.xticks(range(len(avg_scorediff_by_month)), avg_scorediff_by_month.index, rotation=0)
        plt.grid(axis='y', alpha=0.75)
        if save:
            plt.savefig(f'plots/{sport_name}/{sport_name}_mean_for_month.png')
        if plots:
            plt.show()
        plt.close()
        
        logger.info(f"Plots for {sport_name} have been {'saved' if save else 'generated'}.")
        return
    
    def predict_match(self, sport_name: str, match_info: dict):

        allnew = match_info.split(' ')
        if len(allnew) == 4:
            allnew = [allnew[0], allnew[1], (allnew[2] + ' ' + allnew[3])]
        df_row = pd.DataFrame(columns=['date', 'time', 'team2'])
        df_row.loc[0] = allnew
        print(df_row)
        # Process datetime features
        df_row = self.make_datetime_features(df_row)
        print(df_row)
        # Process team features
        df_row = self.make_team_test_features(sport_name, df_row)
        # Select relevant features
        feature_cols = [
            'year', 'day_of_week', 'is_weekend', 'day_of_month',
            'month', 'quarter', 'day_of_year', 'week_of_year', 'season',
            'is_month_start_end', 'hour', 'minute', 'time_of_day',
            'is_work_time', 'decimal_time', 'is_afternoon',
            'days_until_weekend', 'days_until_end_of_month',
            'avg_scorediff_by_team', 'games_played_with_team',
            'cumulative_scorediff_with_team', 'win_ratio_with_team',
            'avg_scorediff_by_season'
        ]
        
        X_new = df_row[feature_cols]
        
        # Load models
        if sport_name == 'hockey':
            path = f'models/hockey/{sport_name}'
        else:
            path = f'models/football/{sport_name}'

        regressor = CatBoostRegressor()
        regressor.load_model(path+"_regressor.cbm", format='cbm')
        classifier = CatBoostClassifier()
        classifier.load_model(path+"_classifier.cbm", format='cbm')
        
        # Make predictions
        score_pred = regressor.predict(X_new)[0]
        win_prob = classifier.predict_proba(X_new)[0][1]
        
        prediction = {
            'score_diff_pred': score_pred,
            'win_probability': win_prob
        }
        
        logger.info(f"Prediction for {sport_name}: {prediction}")
        return prediction
