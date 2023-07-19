import pandas as pd
import math
from matplotlib import pyplot as plt

df = pd.read_excel("scores_by_score_fn.xlsx")

def filter_extrema(df, column_name, score_range):
    filtered = (df[column_name] < score_range[0]) | (df[column_name] > score_range[1])

    tp = (filtered & (df['good_data'] == "No")).sum()
    fp = (filtered & (df['good_data'] == "Yes")).sum()
    fn = (~filtered & (df['good_data'] == "No")).sum()
    total_filtered = filtered.sum()
    filtered_percentage = total_filtered / len(df)

    print(f"# Filtering: {column_name} < {score_range[0]} and {column_name} > {score_range[1]}")
    recall = (tp / (tp + fn))
    precision = (tp / (tp + fp))
    print(f"Recall: {recall:0.3f}")
    print(f"Precision: {precision:0.3f}")
    print(f"Percent filtered: {100 * filtered_percentage:0.3f}%")
    return {
        'precision': precision,
        'recall': recall
    }


filters = [
    {'column_name': 'avg-word-length', 'score_range': [3, 12]},
    {'column_name': 'redundancy', 'score_range': [1.03, 4.5]},
    {'column_name': 'word_frequency', 'score_range': [30, 18000]},
]
for filter in filters:
    # TODO: the same thing but for recall
    # TODO: Worth looking at examples where precision starts to drop to see if we should change their label (FPs)
    # TODO: look into the nans for recall

    # Minimum
    column_values = sorted(df[filter['column_name']].values)
    precision_by_min_threshold = []
    recall_by_min_treshold = []
    for value in column_values:
        filter['score_range'] = [value, math.inf]
        scores = filter_extrema(df, **filter)
        precision_by_min_threshold.append(scores['precision'])
        recall_by_min_treshold.append(scores['recall'])
   
    plt.clf()
    plt.plot(column_values, precision_by_min_threshold)
    plt.xlabel(f"Minimum Threshold for {filter['column_name']}")
    plt.ylabel(f"Precision")
    plt.title(f"Precision x Minimum Threshold for {filter['column_name']}")
    plt.savefig(f"precision_by_minimum_threshold_for_{filter['column_name']}.png")

    plt.clf()
    plt.plot(column_values, recall_by_min_treshold)
    plt.xlabel(f"Minimum Threshold for {filter['column_name']}")
    plt.ylabel(f"recall")
    plt.title(f"recall x Minimum Threshold for {filter['column_name']}")
    plt.savefig(f"recall_by_minimum_threshold_for_{filter['column_name']}.png")

    # Maximum
    column_values = sorted(df[filter['column_name']].values)
    precision_by_max_threshold = []
    recall_by_max_threshold = []
    for value in column_values:
        filter['score_range'] = [-math.inf, value]
        scores = filter_extrema(df, **filter)
        precision_by_max_threshold.append(scores['precision'])
        recall_by_max_threshold.append(scores['recall'])
    plt.clf()
    plt.plot(column_values, precision_by_max_threshold)
    plt.xlabel(f"Maximum Threshold for {filter['column_name']}")
    plt.ylabel(f"Precision")
    plt.title(f"Precision x Maximum Threshold for {filter['column_name']}")
    plt.savefig(f"precision_by_maximum_threshold_for_{filter['column_name']}.png")
    
    plt.clf()
    plt.plot(column_values, recall_by_max_threshold)
    plt.xlabel(f"Maximum Threshold for {filter['column_name']}")
    plt.ylabel(f"Recall")
    plt.title(f"Recall x Maximum Threshold for {filter['column_name']}")
    plt.savefig(f"Recall_by_maximum_threshold_for_{filter['column_name']}.png")
    

    print()

