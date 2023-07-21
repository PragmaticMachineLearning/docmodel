import math
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

FILTERS: list[dict[str, str]] = [
    {"column_name": "avg-word-length"},
    {"column_name": "redundancy"},
    {"column_name": "word_frequency"},
]

df: pd.DataFrame = pd.read_excel("scores_by_score_fn.xlsx")


def filter_extrema(df: pd.DataFrame, column_name: str, score_range: list[float]):
    filtered = (df[column_name] < score_range[0]) | (df[column_name] > score_range[1])

    tp = (filtered & (df["good_data"] == "No")).sum()
    fp = (filtered & (df["good_data"] == "Yes")).sum()
    fn = (~filtered & (df["good_data"] == "No")).sum()
    total_filtered = filtered.sum()
    filtered_percentage = total_filtered / len(df)

    print(
        f"# Filtering: {column_name} < {score_range[0]} and {column_name} > {score_range[1]}"
    )
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    print(f"Recall: {recall:0.3f}")
    print(f"Precision: {precision:0.3f}")
    print(f"Percent filtered: {100 * filtered_percentage:0.3f}%")
    return {"precision": precision, "recall": recall}


def compute_thresholds(df: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    results = {}
    column_names = [filter["column_name"] for filter in FILTERS]

    for column_name in column_names:
        # TODO: the same thing but for recall
        # TODO: Worth looking at examples where precision starts to drop to see if we should change their label (FPs)
        # TODO: look into the nans for recall
        column_values = sorted(df[column_name].values)
        precision_by_min_threshold = []
        precision_by_max_threshold = []
        recall_by_min_treshold = []
        recall_by_max_threshold = []

        for _, value in enumerate(column_values):
            # update score range in for every score range in
            scores_min = filter_extrema(df, column_name, [value, math.inf])
            precision_by_min_threshold.append(scores_min["precision"])
            recall_by_min_treshold.append(scores_min["recall"])

            scores_max = filter_extrema(df, column_name, [-math.inf, value])
            precision_by_max_threshold.append(scores_max["precision"])
            recall_by_max_threshold.append(scores_max["recall"])

        results[column_name] = {
            "precision_min_thresholds": precision_by_min_threshold,
            "precision_max_thresholds": precision_by_max_threshold,
            "recall_min_thresholds": recall_by_min_treshold,
            "recall_max_thresholds": recall_by_max_threshold,
        }
    return results


def plot_thresholds(results: dict[str, dict[str, list[float]]], df: pd.DataFrame)-> None:
    
    output_dir = os.path.join(os.path.dirname(__file__), "threshold_plots")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for column_name, filter_results in results.items():
        column_values = np.sort(df[column_name].values)

        for threshold_type, label in [("min", "Minimum"), ("max", "Maximum")]:
            precision_values = filter_results[f"precision_{threshold_type}_thresholds"]
            recall_values = filter_results[f"recall_{threshold_type}_thresholds"]

            plt.clf()
            plt.plot(column_values, precision_values)
            plt.xlabel(f"{label} Threshold for {column_name}")
            plt.ylabel("Precision")
            plt.title(f"Precision x {label} Threshold for {column_name}")
            plt.savefig(
                f"{output_dir}/precision_by_{threshold_type}_threshold_for_{column_name}.png"
            )

            plt.clf()
            plt.plot(column_values, recall_values)
            plt.xlabel(f"{label} Threshold for {column_name}")
            plt.ylabel("Recall")
            plt.title(f"Recall x {label} Threshold for {column_name}")
            plt.savefig(
                f"{output_dir}/recall_by_{threshold_type}_threshold_for_{column_name}.png"
            )


thresholds: dict[str, dict[str, list[float]]] = compute_thresholds(df)

plot_thresholds(thresholds, df)


# filters = [
#     {'column_name': 'avg-word-length', 'score_range': [3, 12]},
#     {'column_name': 'redundancy', 'score_range': [1.03, 4.5]},
#     {'column_name': 'word_frequency', 'score_range': [30, 18000]},
# ]
# for filter in filters:
#     # TODO: the same thing but for recall
#     # TODO: Worth looking at examples where precision starts to drop to see if we should change their label (FPs)
#     # TODO: look into the nans for recall

#     # Minimum
#     column_values = sorted(df[filter['column_name']].values)
#     precision_by_min_threshold = []
#     recall_by_min_treshold = []
#     for value in column_values:
#         filter['score_range'] = [value, math.inf]
#         scores = filter_extrema(df, **filter)
#         precision_by_min_threshold.append(scores['precision'])
#         recall_by_min_treshold.append(scores['recall'])

#     plt.clf()
#     plt.plot(column_values, precision_by_min_threshold)
#     plt.xlabel(f"Minimum Threshold for {filter['column_name']}")
#     plt.ylabel(f"Precision")
#     plt.title(f"Precision x Minimum Threshold for {filter['column_name']}")
#     plt.savefig(f"precision_by_minimum_threshold_for_{filter['column_name']}.png")

#     plt.clf()
#     plt.plot(column_values, recall_by_min_treshold)
#     plt.xlabel(f"Minimum Threshold for {filter['column_name']}")
#     plt.ylabel(f"recall")
#     plt.title(f"recall x Minimum Threshold for {filter['column_name']}")
#     plt.savefig(f"recall_by_minimum_threshold_for_{filter['column_name']}.png")

#     # Maximum
#     column_values = sorted(df[filter['column_name']].values)
#     precision_by_max_threshold = []
#     recall_by_max_threshold = []
#     for value in column_values:
#         filter['score_range'] = [-math.inf, value]
#         scores = filter_extrema(df, **filter)
#         precision_by_max_threshold.append(scores['precision'])
#         recall_by_max_threshold.append(scores['recall'])
#     plt.clf()
#     plt.plot(column_values, precision_by_max_threshold)
#     plt.xlabel(f"Maximum Threshold for {filter['column_name']}")
#     plt.ylabel(f"Precision")
#     plt.title(f"Precision x Maximum Threshold for {filter['column_name']}")
#     plt.savefig(f"precision_by_maximum_threshold_for_{filter['column_name']}.png")

#     plt.clf()
#     plt.plot(column_values, recall_by_max_threshold)
#     plt.xlabel(f"Maximum Threshold for {filter['column_name']}")
#     plt.ylabel(f"Recall")
#     plt.title(f"Recall x Maximum Threshold for {filter['column_name']}")
#     plt.savefig(f"Recall_by_maximum_threshold_for_{filter['column_name']}.png")
