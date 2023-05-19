import pandas as pd
from knn import calculate_ndcg

ndcg_scores = [(1, 3), (2, 4), (3, 2)]

print(sum(ndcg_scores[1]) / len(ndcg_scores[1]))

df = pd.DataFrame({
    'srch_id': [1, 1, 2, 2, 3, 4],
    'prop_id': [1, 2, 3, 4, 5, 6],
    'target': [3, 2, 3, 4, 5, 2],  # Ground truth rankings
    'predicted_target': [2.8, 2.1, 2.9, 4.2, 5.6, 1.8],  # Predicted rankings
})
df['rank'] = df.groupby('srch_id')['predicted_target'].rank(ascending=False)
df_pred = df.sort_values(['srch_id', 'rank'], ascending=[True, False])
print(df)
print(df_pred[['srch_id', 'prop_id']])

val_ndcg = calculate_ndcg(df_pred, 5)
print(f'The validation NDCG@5 = {val_ndcg}')