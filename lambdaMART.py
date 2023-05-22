import lightgbm as lgb
import optuna

from data import read_processed_train, read_processed_val, read_processed_test

# -------------------------------- READ DATA -------------------------------- #

# Train
train_df = read_processed_train()

# Validation
val_df = read_processed_val()

# TODO: arg = test
# # Test
# test_df = read_processed_test()
# test_df.reset_index(inplace=True)

# ------------------------ ADDITIONAL PREPROCESSING ------------------------- #



# --------------------------- FEATURE ENGINEERING --------------------------- #



# --------------------------------- TUNING ---------------------------------- #
def objective(trial):
    params = {
        # Constant parameters
        "objective": "lambdarank",
        "boosting_type": "gbdt",
        "metric": "ndcg",
        "n_estimators": 1000, 
        "importance_type": "gain",
        "label_gain": [i for i in range(max(y_train.max(), y_val.max()) + 1)],
        "bagging_freq": 1,
        "verbosity": 0,
        # Tuned parameters
        "num_leaves": trial.suggest_int("num_leaves", 2, 2**10),
        'max_depth': trial.suggest_int('max_depth', -1, 500, 50),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.05, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100),
    }

    model = lgb.LGBMRanker(**params)
    model.fit(X_train, y_train, verbose=False)
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions, squared=False)
    return rmse


# -------------------------------- TRAINING --------------------------------- #


# ---------------------------------- MAIN ----------------------------------- #

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)

    print('Best hyperparameters:', study.best_params)
    print('Best RMSE:', study.best_value)

