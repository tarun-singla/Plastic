import numpy as np
import csv

def import_data():
    X = np.genfromtxt("train_X_knn.csv", delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_knn.csv", delimiter=',', dtype=np.float64)
    return X, Y

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
      top_k_nn_indices = find_k_nearest_neighbors(train_X, test_elem_x, k,n)
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

def calculate_accuracy(predicted_Y, actual_Y):
    total_num_of_observations = len(predicted_Y)
    num_of_values_matched = 0
    for i in range(0,total_num_of_observations):
        if(predicted_Y[i] == actual_Y[i]):
            num_of_values_matched +=1
    return float(num_of_values_matched)/total_num_of_observations

def get_best_k_using_validation_set(train_X, train_Y, validation_split_percent,n):
    import math
    total_num_of_observations = len(train_X)
    train_length = math.floor((float(100 - validation_split_percent))/100 * total_num_of_observations )
    validation_X = train_X[train_length:]
    validation_Y = train_Y[train_length:]
    train_X = train_X[0:train_length]
    train_Y = train_Y[0:train_length]

    best_k = -1
    best_accuracy = 0
    for k in range(1, train_length+1):
        predicted_Y = classify_points_using_knn(train_X,train_Y,validation_X, n,k)
        accuracy = calculate_accuracy(predicted_Y,validation_Y)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

    return best_k

def train_model(X, Y):
    W = get_best_k_using_validation_set(X, Y, 70, 7)
    best_k = [[W]]
    return best_k

def save_model(weights, weights_file_name):
    with open(weights_file_name, 'w') as weights_file:
        wr = csv.writer(weights_file)
        wr.writerows(weights)
        weights_file.close()

if __name__ == "__main__":
    X, Y = import_data()
    weights = train_model(X, Y)
    save_model(weights, "WEIGHTS_FILE.csv")
