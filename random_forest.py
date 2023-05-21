from sklearn.ensemble import RandomForestRegressor

from data import read_processed_train, save_model


def train_model():
    print('Reading training data')
    data = read_processed_train()
    data = data.set_index(['srch_id', 'prop_id'], drop=True)
    print(data.shape)

    feature_names = list(data.columns)
    feature_names.remove('date_time')
    feature_names.remove('position')
    feature_names.remove('click_bool')
    feature_names.remove('booking_bool')
    feature_names.remove('target')

    print(f'Features: {feature_names}')
    features = data[feature_names].values
    target = data.target.values

    print("Training the model")
    model = RandomForestRegressor(n_estimators=70, verbose=2, n_jobs=2, min_samples_split=10, random_state=42)
    model.fit(features, target)

    print("Saving the model")
    save_model(model)


if __name__ == '__main__':
    train_model()
