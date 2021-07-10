import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

cbc = CatBoostClassifier()
cbc.load_model("model")
df = pd.read_csv('train.csv')

df['target'] = 0
df['target'][df['full_time_home_goals'] > df['full_time_away_goals']] = 1
df['target'][df['full_time_home_goals'] < df['full_time_away_goals']] = -1

Mean_enc_div = df.groupby(['Division'])['target'].mean().to_dict()
Mean_enc_home = df.groupby(['home_team'])['target'].mean().to_dict()
Mean_enc_away = df.groupby(['away_team'])['target'].mean().to_dict()

for i in range(int(input())):
    match = pd.DataFrame(np.array(input().split()).astype(float).reshape((1, 8)),
                         columns=['Division', 'Time', 'home_team', 'away_team',
                                  'Referee', 'home_coef', 'draw_coef', 'away_coef'])

    match = match.drop(['Time'], axis=1)

    match['Division_mean'] = match['Division'].apply(lambda x: Mean_enc_div[x] if x in Mean_enc_div.keys() else 0)
    match['home_team_mean'] = match['home_team'].apply(lambda x: Mean_enc_home[x] if x in Mean_enc_home.keys() else 0)
    match['away_team_mean'] = match['away_team'].apply(lambda x: Mean_enc_away[x] if x in Mean_enc_away.keys() else 0)

    match[match == -1] = np.nan
    match['buck_pred'] = np.argmin([match['away_coef'], match['draw_coef'], match['home_coef']], axis=0) - 1

    pred = cbc.predict_proba(match)

    if pred[0][0] >= 0.5:
        print('HOME', flush=True)
    elif pred[0][1] >= 0.5:
        print('DRAW', flush=True)
    elif pred[0][2] >= 0.5:
        print('AWAY', flush=True)
    else:
        print('SKIP', flush=True)

    _ = input()
