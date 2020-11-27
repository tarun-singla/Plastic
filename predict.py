import numpy as np
import csv
import sys

# from validate import validate

"""
Predicts the target values for data in the file at 'test_X_file_path'.
Writes the predicted values to the file named "predicted_test_Y_knn.csv".
"""

def import_data(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64, skip_header=1)
    return test_X

def import_data2(test_X_file_path):
    test_X = np.genfromtxt(test_X_file_path, delimiter=',', dtype=np.float64)
    return test_X

def compute_ln_norm_distance(vector1, vector2, n):
    vector_len = len(vector1)
    distance = 0
    for i in range(0, vector_len):
      diff = (abs(vector1[i] - vector2[i]))**n
      distance += diff
    distance = distance ** (1.0/n)
    return distance

def find_k_nearest_neighbors(train_X, test_example, k, n):
    indices_dist_pairs = []
    index= 0
    for train_elem_x in train_X:
      distance = compute_ln_norm_distance(train_elem_x, test_example, n)
      indices_dist_pairs.append([index, distance])
      index += 1
    indices_dist_pairs.sort(key = lambda x: x[1])
    top_k_pairs = indices_dist_pairs[:k]
    top_k_indices = [i[0] for i in top_k_pairs]
    return top_k_indices

def classify_points_using_knn(train_X, train_Y, test_X, k, n):
    test_Y = []
    for test_elem_x in test_X:
      top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k, n)
      top_knn_labels = []

      for i in top_k_nn_indices:
        top_knn_labels.append(train_Y[i])
      Y_values = list(set(top_knn_labels))

      max_count = 0
      most_frequent_label = -1
      for y in Y_values:
        count = top_knn_labels.count(y)
        if(count > max_count):
          max_count = count
          most_frequent_label = y
          
      test_Y.append(most_frequent_label)
    return test_Y

def predict_target_values(test_X):
    best_k = int(import_data2("WEIGHTS_FILE.csv"))
    train_X = import_data("train_X_knn.csv")
    train_Y = import_data2("train_Y_knn.csv")
    test_Y = classify_points_using_knn(train_X, train_Y, test_X, best_k, 7)
    return test_Y

def write_to_csv_file(pred_Y, predicted_Y_file_name):
    with open(predicted_Y_file_name, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file)
        wr.writerows(map(lambda x: [x], pred_Y))
        csv_file.close()


def predict(test_X_file_path):
    test_X = import_data(test_X_file_path)
    pred_Y = predict_target_values(test_X)
    write_to_csv_file(pred_Y, "predicted_test_Y_knn.csv")


if __name__ == "__main__":
    test_X_file_path = sys.argv[1]
    predict(test_X_file_path)
    # Uncomment to test on the training data
    # validate(test_X_file_path, actual_test_Y_file_path="train_Y_knn.csv") 
