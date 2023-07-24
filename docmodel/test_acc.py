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
    filtered = (df[column_name] <= score_range[0]) | (df[column_name] >= score_range[1])

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
    column_names: list[str] = [filter["column_name"] for filter in FILTERS]

    for column_name in column_names:
        # TODO: the same thing but for recall
        # TODO: Worth looking at examples where precision starts to drop to see if we should change their label (FPs)
        # TODO: look into the nans for recall
        column_values: list[int] = sorted(df[column_name].values)
        precision_by_min_threshold = []
        precision_by_max_threshold = []
        recall_by_min_treshold = []
        recall_by_max_threshold = []
        coverage_by_min_threshold = []
        coverage_by_max_threshold = []

        for idx , value in enumerate(column_values):
            # update score range in for every score range in
            scores_min = filter_extrema(df, column_name, [value, math.inf])
            coverage_for_min = (idx+1)/len(column_values) 
            coverage_by_min_threshold.append(coverage_for_min)
            precision_by_min_threshold.append(scores_min["precision"])
            recall_by_min_treshold.append(scores_min["recall"])

            scores_max = filter_extrema(df, column_name, [-math.inf, value])
            coverage_for_max = 1 - (idx/len(column_values))
            coverage_by_max_threshold.append(coverage_for_max)
            precision_by_max_threshold.append(scores_max["precision"])
            recall_by_max_threshold.append(scores_max["recall"])

        results[column_name] = {
            "precision_min_thresholds": precision_by_min_threshold,
            "precision_max_thresholds": precision_by_max_threshold,
            "recall_min_thresholds": recall_by_min_treshold,
            "recall_max_thresholds": recall_by_max_threshold,
            "coverage_min_thresholds": coverage_by_min_threshold,
            "coverage_max_thresholds": coverage_by_max_threshold
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
            coverage_values = filter_results[f"coverage_{threshold_type}_thresholds"]
            
            plt.clf()
            fig, ax1 = plt.subplots()

            color = 'tab:blue'
            ax1.set_xlabel(f"{label} Threshold for {column_name}")
            ax1.set_ylabel('Precision', color=color)
            ax1.plot(column_values, precision_values, color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.set_ylabel('Recall', color=color)  # we already handled the x-label with ax1
            ax2.plot(column_values, recall_values, color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.show()
            # plt.savefig(
            #     f"{output_dir}/precision_by_{threshold_type}_threshold_for_{column_name}.png"
            # )

            # plt.clf()
            # fig, ax1 = plt.subplots()

            # color = 'tab:blue'
            # ax1.set_xlabel(f"{label} Threshold for {column_name}")
            # ax1.set_ylabel('Recall', color=color)
            # ax1.plot(column_values, recall_values, color=color)
            # ax1.tick_params(axis='y', labelcolor=color)

            # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            # color = 'tab:red'
            # ax2.set_ylabel('Coverage', color=color)  # we already handled the x-label with ax1
            # ax2.plot(column_values, coverage_values, color=color)
            # ax2.tick_params(axis='y', labelcolor=color)

            # fig.tight_layout()  # otherwise the right y-label is slightly clipped
            # plt.show()
            # plt.savefig(
            #     f"{output_dir}/recall_by_{threshold_type}_threshold_for_{column_name}.png"
            # )
            
            # plt.clf()
            # plt.plot(column_values, precision_values)
            # plt.xlabel(f"{label} Threshold for {column_name}")
            # plt.ylabel("Precision")
            # plt.title(f"Precision x {label} Threshold for {column_name}")
            # plt.savefig(
            #     f"{output_dir}/precision_by_{threshold_type}_threshold_for_{column_name}.png"
            # )

            # plt.clf()
            # plt.plot(column_values, recall_values)
            # plt.xlabel(f"{label} Threshold for {column_name}")
            # plt.ylabel("Recall")
            # plt.title(f"Recall x {label} Threshold for {column_name}")
            # plt.savefig(
            #     f"{output_dir}/recall_by_{threshold_type}_threshold_for_{column_name}.png"
            # )


thresholds: dict[str, dict[str, list[float]]] = compute_thresholds(df)

plot_thresholds(thresholds, df)


