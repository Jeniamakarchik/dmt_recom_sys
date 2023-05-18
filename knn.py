import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# don't group per search id 
def knn_ranking(df, k=3):

    # use cosine similarity
    model = NearestNeighbors(n_neighbors=k, metric='cosine')

    # List to store the ranking data
    ranking_data = []
    
    # 
    for srch_id, group in df.groupby('srch_id'):

        # Fit the model using the group data
        # maybe leave prop id in 
        features = group.drop(['srch_id', 'prop_id', 'target', 'click_bool', 'booking_bool', 'position', 'date_time'], axis=1).values
        model.fit(features)

        for index, row in group.iterrows():

            # Predict the target for each row in the group
            distances, indices = model.kneighbors([row.drop(['srch_id', 'prop_id', 'target']).values])
            neighbor_ranks = group.iloc[indices[0]]['target'].values
            score = np.mean(neighbor_ranks)

            # Append the srch_id, prop_id, and score to the ranking data
            ranking_data.append((srch_id, row['prop_id']), score)

    # Convert the ranking data to a DataFrame
    ranking_df = pd.DataFrame(ranking_data, columns=['srch_id', 'prop_id'])

    return ranking_df


# make train function


