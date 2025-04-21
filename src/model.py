import pandas as pd
from xgboost import XGBClassifier

class MatchPredictor:
    def __init__(self):
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def train(self, df):
        X = df.drop('winner', axis=1)
        y = df['winner']
        self.model.fit(X, y)

    def predict(self, p1, p2):
        features = [
            p1['elo'] - p2['elo'],
            p1['ranking'] - p2['ranking'],
            p1['clay_win_pct'] - p2['clay_win_pct']
        ]
        return 1 if self.model.predict([features])[0] == 1 else 2
