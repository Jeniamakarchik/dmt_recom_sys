import lightgbm as lgb
import optuna

from data import read_processed_train, read_processed_val, read_processed_test

# -------------------------------- READ DATA -------------------------------- #

# Train
train_df = read_processed_train()
train_df.reset_index(inplace=True)

# Validation
val_df = read_processed_val()
val_df.reset_index(inplace=True)

# TODO: arg = test
# # Test
# test_df = read_processed_test()
# test_df.reset_index(inplace=True)

# -------------------------------- Tuning -------------------------------- #


