from data import read_processed_knn_test
from knn import calculate_ndcg
import pandas as pd
import numpy as np

# Load and shuffle data
test_data = read_processed_knn_test()
#test_data = test_data.sample(frac=1).reset_index(drop=True)

# Add random targets
test_data['target'] = np.random.uniform(0, 5, test_data.shape[0])
test_data['predicted_target'] = np.random.uniform(0, 5, test_data.shape[0])

print(test_data.head())

# Calculate NDCG@5 for random ranking
ndcg = calculate_ndcg(test_data, 5)
print(f'The NDCG@% score for random ranking is {ndcg}')




