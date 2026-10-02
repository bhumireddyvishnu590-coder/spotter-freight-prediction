"""
Spotter Machine Learning Engineer Assessment
Freight Rate Prediction Pipeline
Candidate: Vishnuvardhan Reddy Bhumireddy
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_DIR = Path("data") if Path("data").exists() else Path(".")

def load_data():
    train_path = DATA_DIR / "train-test.csv"
    val_path = DATA_DIR / "validation.csv"
    dec_path = DATA_DIR / "december-chart-inputs.csv"
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    dec_df = pd.read_csv(dec_path)
    return train_df, val_df, dec_df

def feature_engineering(train_df, val_df, dec_df):
    train = train_df.copy()
    val = val_df.copy()
    dec = dec_df.copy()

    # 1. Clean cargo weight: correct negative values (sign-inversion typos)
    train['weight'] = train['weight'].abs()
    val['weight'] = val['weight'].abs()
    dec['weight'] = dec['weight'].abs()

    median_weight = train['weight'].median()
    train['weight'] = train['weight'].fillna(median_weight)
    val['weight'] = val['weight'].fillna(median_weight)
    dec['weight'] = dec['weight'].fillna(median_weight)

    # 2. Impute missing market index
    median_market = train['market_index'].median()
    train['market_index'] = train['market_index'].fillna(median_market)
    val['market_index'] = val['market_index'].fillna(median_market)

    # 3. Fix coordinates for the fixed December scenario (Lexington -> Fort Wayne)
    lex_coords = train[train['pickup'] == 'Lexington'][['pickup_lat', 'pickup_lon']].iloc[0]
    fw_coords = train[train['delivery'] == 'Fort Wayne'][['delivery_lat', 'delivery_lon']].iloc[0]

    dec['pickup_lat'] = lex_coords['pickup_lat']
    dec['pickup_lon'] = lex_coords['pickup_lon']
    dec['delivery_lat'] = fw_coords['delivery_lat']
    dec['delivery_lon'] = fw_coords['delivery_lon']

    # Impute December daily market quote signal from validation December data
    val_dec = val[val['date'].between('2025-12-01', '2025-12-31')]
    daily_dec = val_dec.groupby('date')[['market_index', 'quote_signal']].mean().reset_index()
    dec = dec.merge(daily_dec, on='date', how='left')

    datasets = [train, val, dec]
    for df in datasets:
        # Date & Temporal features
        dt = pd.to_datetime(df['date'])
        df['day_of_week'] = dt.dt.dayofweek
        df['day_of_month'] = dt.dt.day
        df['month'] = dt.dt.month
        df['day_of_year'] = dt.dt.dayofyear
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Domain freight pricing signals
        df['signal_rate_baseline'] = df['distance'] * df['quote_signal']
        df['market_signal_interaction'] = df['market_index'] * df['quote_signal']
        df['weight_per_mile'] = df['weight'] / (df['distance'] + 1e-5)
        df['lat_diff'] = (df['delivery_lat'] - df['pickup_lat']).abs()
        df['lon_diff'] = (df['delivery_lon'] - df['pickup_lon']).abs()
        df['manhattan_dist'] = df['lat_diff'] + df['lon_diff']
        df['route'] = df['pickup'] + "__" + df['delivery']

    # Target rate & frequency encoding for high-cardinality routes
    route_counts = train['route'].value_counts()
    route_mean_rate = train.groupby('route')['posted_rate'].mean()
    global_mean_rate = train['posted_rate'].mean()

    for df in datasets:
        df['route_freq'] = df['route'].map(route_counts).fillna(1)
        df['route_target_enc'] = df['route'].map(route_mean_rate).fillna(global_mean_rate)

    return datasets[0], datasets[1], datasets[2]

def run_pipeline():
    print("Loading data...")
    train_df, val_df, dec_df = load_data()

    print("Engineering features...")
    train, val, dec = feature_engineering(train_df, val_df, dec_df)

    categorical_cols = ['equipment', 'pickup', 'delivery']
    feature_cols = [
        'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon', 'distance',
        'equipment', 'weight', 'day_of_week', 'day_of_month', 'month', 'day_of_year',
        'is_weekend', 'market_index', 'quote_signal', 'signal_rate_baseline',
        'market_signal_interaction', 'weight_per_mile', 'lat_diff', 'lon_diff',
        'manhattan_dist', 'pickup', 'delivery', 'route_freq', 'route_target_enc'
    ]

    cat_indices = [feature_cols.index(c) for c in categorical_cols]

    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    train[categorical_cols] = encoder.fit_transform(train[categorical_cols].astype(str))
    val[categorical_cols] = encoder.transform(val[categorical_cols].astype(str))
    dec[categorical_cols] = encoder.transform(dec[categorical_cols].astype(str))

    X_train = train[feature_cols]
    y_train = train['posted_rate']
    X_val = val[feature_cols]
    X_dec = dec[feature_cols]

    print("Training 5-Fold Cross Validation Models...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []
    cv_maes, cv_rmses, cv_r2s = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va, y_va = X_train.iloc[val_idx], y_train.iloc[val_idx]

        model = HistGradientBoostingRegressor(
            max_iter=450,
            learning_rate=0.05,
            max_leaf_nodes=63,
            categorical_features=cat_indices,
            random_state=42 + fold
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        cv_maes.append(mean_absolute_error(y_va, preds))
        cv_rmses.append(np.sqrt(mean_squared_error(y_va, preds)))
        cv_r2s.append(r2_score(y_va, preds))
        models.append(model)

    print(f"5-Fold CV -> MAE: ${np.mean(cv_maes):.2f} | RMSE: ${np.mean(cv_rmses):.2f} | R2: {np.mean(cv_r2s):.4f}")

    print("Training Full Model for Ensemble...")
    full_model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_leaf_nodes=63,
        categorical_features=cat_indices,
        random_state=100
    )
    full_model.fit(X_train, y_train)

    val_preds = (full_model.predict(X_val) + np.mean([m.predict(X_val) for m in models], axis=0)) / 2.0
    dec_preds = (full_model.predict(X_dec) + np.mean([m.predict(X_dec) for m in models], axis=0)) / 2.0

    val_preds = np.clip(val_preds, a_min=10.0, a_max=None)
    dec_preds = np.clip(dec_preds, a_min=10.0, a_max=None)

    # 1. Output validation_predictions.csv
    submission_val = pd.DataFrame({
        'load_id': val['load_id'],
        'predicted_rate': np.round(val_preds, 2)
    })
    submission_val.to_csv('validation_predictions.csv', index=False)
    print("Saved 'validation_predictions.csv' (12,000 rows)")

    # 2. Output data/december_chart_inputs.csv
    dec_out = dec_df.copy()
    dec_out['predicted_rate'] = np.round(dec_preds, 2)
    Path('data').mkdir(exist_ok=True)
    dec_out.to_csv(Path('data') / 'december_chart_inputs.csv', index=False)
    dec_out.to_csv('december_predictions.csv', index=False)
    print("Saved 'data/december_chart_inputs.csv' (31 rows)")

if __name__ == '__main__':
    run_pipeline()