### All DT Helper functions

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import random
from pprint import pprint
from collections import Counter

### Preprocess data - "trips" variable
def assign_trip(hp, dest):
    
    trip_val = ''
    
    if hp == 'Earth' :
        if dest == 'TRAPPIST-1e':
            trip_val = 'A'
        elif dest == '55 Cancri e':
            trip_val = 'B'
        else:
            trip_val = 'C'
    elif hp == 'Europa':
        if dest == 'TRAPPIST-1e':
            trip_val = 'D'
        elif dest == '55 Cancri e':
            trip_val = 'E'
        else:
            trip_val = 'F'
    elif hp == 'Mars':
        if dest == 'TRAPPIST-1e':
            trip_val = 'G'
        elif dest == '55 Cancri e':
            trip_val = 'H'
        else:
            trip_val = 'I'
            
    return trip_val

### Preprocess data - whole flow
def data_preprocess(data):
    ## rename label
    data['label'] = data.Transported
    
    ## Handle nulls 
    med_age = data.Age.median()
    mode_HP = data.HomePlanet.mode()[0]
    mode_CS = data.CryoSleep.mode()[0]
    mode_dest = data.Destination.mode()[0]
    data = data.fillna({'HomePlanet' : mode_HP,
                    'CryoSleep' : mode_CS,
                    'Destination' : mode_dest,
                    'Age' : med_age,
                    'RoomService' : 0,
                    'FoodCourt' : 0,
                    'ShoppingMall' : 0,
                    'Spa' : 0,
                    'VRDeck' : 0})
    
    ## Create Exp column 
    data['Exp'] = data.RoomService + data.FoodCourt + data.ShoppingMall + data.Spa + data.VRDeck
    
    ## Create trip column
    for i, row in data.iterrows():
        data.loc[i, 'Trip'] = assign_trip(row['HomePlanet'], row['Destination'])
        
    ## Fix value spaces 
    data['Destination'] = data['Destination'].replace('55 Cancri e', '55_Cancri_e')
    data['Destination'] = data['Destination'].replace('PSO J318.5-22', 'PSO_J318.5-22 ')
    
    ## Create two approaches
    approach_1 = ['CryoSleep', 'Age', 'VIP', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Trip', 'label']
    approach_2 = ['HomePlanet', 'CryoSleep','Destination', 'Age','VIP',  'Exp', 'label']
    
    ## Create dataframes
    app_1_df = data[approach_1].copy()
    app_2_df = data[approach_2].copy()
    
    return app_1_df, app_2_df

### Train Test Split
def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df

### Data Purity
def check_purity(data):
    
    label_column = data[:, -1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False
    
### Classification
def classify_data(data):
    
    label_column = data[:, -1]
    unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)

    index = counts_unique_classes.argmax()
    classification = unique_classes[index]
    
    return classification

### Draw bootstrap
def draw_bootstrap(data):
    bootstrap_indices = list(np.random.choice(range(len(data)), len(data), replace = True))
    oob_indices = [i for i in range(len(data)) if i not in bootstrap_indices]
    
    data_bootstrap = data.iloc[bootstrap_indices]
    
    data_oob = data.iloc[oob_indices]
    
    return data_bootstrap, data_oob

### Feature sampling 
def feature_scramble(data, max_features):
    
    feature_ls = list()
    num_features = list(data.shape)[1] - 2
    
    while len(feature_ls) <= max_features:
        feature_idx = random.sample(range(num_features), 1)
        if feature_idx not in feature_ls:
            feature_ls.extend(feature_idx)
    
    unique = []
    
    for val in feature_ls:
        if val not in unique:
            unique.append(val)
    
    return unique

### Potential Splits
def get_potential_splits(data, max_features):
    
    
    potential_splits = {}
    target_features = feature_scramble(data, max_features=max_features)
    
    for column_index in target_features:          # excluding the last column which is the label
        values = data[:, column_index]
        unique_values = np.unique(values)
        
        potential_splits[column_index] = unique_values
    
    return potential_splits

### Split Data
def split_data(data, split_column, split_value):
    
    split_column_values = data[:, split_column]

    type_of_feature = FEATURE_TYPES[split_column]
    if type_of_feature == "continuous":
        data_below = data[split_column_values <= split_value]
        data_above = data[split_column_values >  split_value]
    
    # feature is categorical   
    else:
        data_below = data[split_column_values == split_value]
        data_above = data[split_column_values != split_value]
    
    return data_below, data_above

### Calculate Entropy
def calculate_entropy(data):
    
    label_column = data[:, -1]
    _, counts = np.unique(label_column, return_counts=True)

    probabilities = counts / counts.sum()
    entropy = sum(probabilities * -np.log2(probabilities))
     
    return entropy

### Lowest Overall Entropy
def calculate_overall_entropy(data_below, data_above):
    
    n = len(data_below) + len(data_above)
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    overall_entropy =  (p_data_below * calculate_entropy(data_below) 
                      + p_data_above * calculate_entropy(data_above))
    
    return overall_entropy

### information gain
def calculate_info_gain(data_unsplit, data_below, data_above):
    
    unsplit_entropy = calculate_entropy(data_unsplit)
    overall = calculate_overall_entropy(data_below, data_above)
    
    return unsplit_entropy - overall

### Determine Best Split
def determine_best_split(data, potential_splits):
    
    best_split_column = None
    best_split_value = None
    
    overall_IG = -999
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below, data_above = split_data(data, split_column=column_index, split_value=value)
            current_IG = calculate_info_gain(data, data_below, data_above)

            if current_IG >= overall_IG:
                overall_IG = current_IG
                best_split_column = column_index
                best_split_value = value
    
    return best_split_column, best_split_value

### Determine Feature Type
def determine_type_of_feature(df):
    
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types

### Decision Tree
def decision_tree_algorithm(df, counter, min_samples, max_depth, max_features): 
    
    # data preparations
    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        data = df.values
    else:
        data = df           
    
    
    # base cases
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        classification = classify_data(data)
        
        return classification

    
    # recursive part
    else:    
        counter += 1

        # helper functions 
        potential_splits = get_potential_splits(data, max_features)
        split_column, split_value = determine_best_split(data, potential_splits)
        data_below, data_above = split_data(data, split_column, split_value)
        
        # check for empty data
        if len(data_below) == 0 or len(data_above) == 0:
            classification = classify_data(data)
            return classification
        
        # determine question
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPES[split_column]
        if type_of_feature == "continuous":
            question = "{} <= {}".format(feature_name, split_value)
            
        # feature is categorical
        else:
            question = "{} = {}".format(feature_name, split_value)
        
        # instantiate sub-tree
        sub_tree = {question: []}
        
        # find answers (recursion)
        yes_answer = decision_tree_algorithm(data_below, counter, min_samples, max_depth, max_features)
        no_answer = decision_tree_algorithm(data_above, counter, min_samples, max_depth, max_features)
        
        # If the answers are the same, then there is no point in asking the qestion.
        # This could happen when the data is classified even though it is not pure
        # yet (min_samples or max_depth base case).
        if yes_answer == no_answer:
            sub_tree = yes_answer
        else:
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree
    
### Classify an Example
def classify_example(example, tree):
    question = list(tree.keys())[0]
    # print(question)
    feature_name, comparison_operator, value = question.split(" ")

    # ask question
    if comparison_operator == "<=":  # feature is continuous
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else:
            answer = tree[question][1]
    
    # feature is categorical
    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else:
            answer = tree[question][1]

    # base case
    if not isinstance(answer, dict):
        return answer
    
    # recursive part
    else:
        residual_tree = answer
        return classify_example(example, residual_tree)
    
### Calculate Model Accuracy
def calculate_accuracy(df, tree):
    
    df["classification"] = df.apply(classify_example, axis=1, args=(tree,))
    df["classification_correct"] = df["classification"] == df["label"]
    
    accuracy = df["classification_correct"].mean()
    
    return accuracy

### Execute Model
def run_model(df, test_size=0.2, max_depth=5):
    train_df, test_df = train_test_split(df, test_size=test_size)
    tree = decision_tree_algorithm(train_df, max_depth=max_depth)
    accuracy = calculate_accuracy(test_df, tree)
    return accuracy
    
### Random Forest 
def random_forest(data, n_estimators, max_features=5, max_depth=5, min_samples=2):
    
    tree_ls = list()
    oob_ls = list()
    
    for i in range(n_estimators):
        data_boot, data_oob = draw_bootstrap(data)
        tree = decision_tree_algorithm(data_boot, counter=0, min_samples=min_samples, max_depth=max_depth, max_features=max_features)
        tree_ls.append(tree)
        oob_error = 1 - calculate_accuracy(data_oob, tree)
        oob_ls.append(oob_error)
    
    print(np.mean(oob_ls))
    
    return tree_ls

### Predict a single instance 
def predict(val, trees):
    
    votes = []
    
    for tree in trees:
        votes.append(classify_example(val, tree))
    
    counter = Counter(votes)
    
    return counter.most_common(1)[0][0]

### Create prediction series 
def prediction(data, trees):
    
    predictions = []
    
    for i in range(data.shape[0]):
        result = predict(data.iloc[i], trees)
        predictions.append(result)
    
    return predictions

### Evaluate accuracy of random forest 
def evaluate(data, predictions, label='label'):
    data['prediction'] = predictions 
    data['correct'] = data['prediction'] == data[label]
    return data['correct'].mean()

### Evaluate RF model with data 
def model_testing(data, trees):
    predictions = prediction(data, trees)
    acc = evaluate(data, predictions)
    
    return acc

### Create indices for cross validation 
def kfold_indices(data, k):
    fold_size = len(data) // k
    indices = np.arange(len(data))
    folds = []
    for i in range(k):
        test_indices = indices[i * fold_size: (i + 1) * fold_size]
        train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        folds.append((train_indices, test_indices))
    return folds

### Get cross val scores
def cross_val(data, k, estimators):
    
    scores = []
    
    fold_indices = kfold_indices(data, k)
    
    for train_indices, test_indices in fold_indices:
        data_train = data.iloc[train_indices]
        data_test = data.iloc[test_indices]
        
        trees = random_forest(data_train, n_estimators=estimators)
        fold_score = model_testing(data_test, trees)
        
        scores.append(fold_score)
    
    print(f'mean accuracy score: {np.mean(scores)}')
    
    return scores