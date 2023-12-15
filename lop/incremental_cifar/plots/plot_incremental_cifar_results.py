# built-in
import os
import argparse

# third party libraries
import matplotlib.pyplot as plt
import numpy as np
from mlproj_manager.plots_and_summaries.plotting_functions import line_plot_with_error_bars, lighten_color


def get_max_over_bins(np_array, bin_size: int):
    """
    Gets the max over windows of size bin_size
    """
    num_bins = np_array.size // bin_size
    reshaped_array = np_array.reshape(num_bins, bin_size)
    return np.max(reshaped_array, axis=1)


def get_min_over_bins(np_array, bin_size: int):
    """
    Gets the min over windows of size of bin_size
    """
    num_bins = np_array.size // bin_size
    reshaped_array = np_array.reshape(num_bins, bin_size)
    return np.min(reshaped_array, axis=1)


def line_plot_with_shaded_region(average, standard_error, color, label):
    """
    Creates a line plot with shaded regions
    """
    line_plot_with_error_bars(results=average, error=standard_error, color=color, x_axis=np.arange(average.size) + 1,
                              light_color=lighten_color(color, 0.1), label=label)


def get_colors(algorithms: list):
    """
    Returns a dictionary of colors. The 5 algorithms in the paper have predetermined colors. Any other named algorithm
    gets assigned a random color

    param algorithms: list of names of algorithms
    return: dictionary of name-color pairs
    """

    pre_assigned_colors = {
        "base_deep_learning_system": "#d62728",     # tab: red
        "retrained_network": "#7f7f7f",             # tab: grey
        "head_resetting": "#2ca02c",                # tab: green
        "shrink_and_perturb": "#ff7f0e",            # tab: orange
        "continual_backpropagation": "#1f77b4",            # tab: blue
    }

    other_colors = ["#FBB829",  # heart of gold
                    "#ADD8C7",  # old pea green
                    "#E51959",  # blushing emoji pink
                    "#5A395F",  # grimace purple
                    "#813E13"  # like, a really brown horse
                    ]

    color_index = 0
    actual_colors = {}
    for alg in algorithms:
        if alg in pre_assigned_colors.keys():
            actual_colors[alg] = pre_assigned_colors[alg]
        else:
            if color_index == len(other_colors):
                raise ValueError("Not enough colors!")
            actual_colors[alg] = other_colors[color_index]
            color_index += 1
    return actual_colors


def retrieve_results(algorithms: list, metric: str, results_dir: str):
    """
    Loads into memory all the results data corresponding to each algorithm and the given metric

    :param algorithms: list of strings corresponding to algorithm names
    :param metric: string corresponding to one of the metrics
    :param results_dir: path to directory containing all the name experiments results
    return: dictionary of algorithm names - numpy array pairs
    """

    if metric == "relative_accuracy_per_epoch":
        metric = "test_accuracy_per_epoch"

    results_dict = {}
    total_num_epochs = 4000
    epochs_per_task = 200
    denominator = 512 if "rank" in metric else 1.0
    start_idx = 1 if "next" in metric else 0

    for alg in algorithms:
        temp_dir = os.path.join(results_dir, alg, metric)
        num_samples = len(os.listdir(temp_dir))
        temp_results = np.zeros((num_samples, total_num_epochs // epochs_per_task), dtype=np.float32)

        for index in range(num_samples):
            temp_result_path = os.path.join(temp_dir, "index-{0}.npy".format(index))
            index_results = np.load(temp_result_path) / denominator

            if "accuracy" in metric:
                index_results = get_max_over_bins(index_results, bin_size=epochs_per_task)
            elif "loss" in metric:
                index_results = get_min_over_bins(index_results, bin_size=epochs_per_task)
            else:
                pass

            temp_results[index] = index_results

        results_dict[alg] = temp_results[start_idx:, :]

    return results_dict


def plot_all_results(results_dict: dict, colors: dict, metric: str):
    """
    Makes a line plot for each different algorithm in results dict

    :param results_dict: dictionary of (algorithm names, results) pairs
    :param colors: dictionary of (algorithm names, color hex key) pairs
    :param metric: str corresponding to the metric being plotted
    """

    if metric == "relative_accuracy_per_epoch":
        assert "retrained_network" in results_dict.keys()

    fig, ax = plt.subplots()
    for alg, results in results_dict.items():

        if metric == "relative_accuracy_per_epoch":
            if alg == "retrained_network":
                continue
            else:
                num_samples = results.shape[0]
                if results_dict["retrained_network"].shape[0] < num_samples:
                    raise ValueError("There are not enough samples for the baseline")
                results = results - results_dict["retrained_network"][:num_samples, :]

        results_mean = np.average(results, axis=0)
        results_std = np.zeros_like(results_mean)
        num_samples = results.shape[0]
        if num_samples > 1:
            results_std = np.std(results, axis=0, ddof=1) / np.sqrt(num_samples)

        line_plot_with_shaded_region(results_mean, results_std, colors[alg], label=alg)

    ax.yaxis.grid()


def create_plots(plot_arguments: dict):
    """
    Creates a dictionary with colors.
    """
    algorithms = plot_arguments["algorithms"].split(",")
    metric = plot_arguments["metric"]
    results_dir = plot_arguments["results_dir"]

    colors = get_colors(algorithms)
    results = retrieve_results(algorithms, metric, results_dir)

    plot_all_results(results, colors, metric)

    plt.ylabel(metric)
    plt.xlabel("Task Number")
    plt.legend()
    file_path = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(file_path, metric + ".svg"), dpi=200)


def main():

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", action="store", type=str, default="./lop/incremental_cifar/results/",
                        help="Path to directory containing the results of all the named experiments.")
    parser.add_argument("--algorithms", action="store", type=str, default="base_deep_learning_system",
                        help="Comma separated list of algorithms.")
    parser.add_argument("--metric", action="store", type=str, default="test_accuracy_per_epoch",
                        help="Metric to plot for each algorithm.",
                        choices=["next_task_dormant_units_analysis", "relative_accuracy_per_epoch",
                                 "next_task_effective_rank_analysis", "next_task_stable_rank_analysis",
                                 "previous_tasks_dormant_units_analysis", "previous_tasks_effective_rank_analysis",
                                 "previous_tasks_stable_rank_analysis", "test_accuracy_per_epoch",
                                 "test_loss_per_epoch", "weight_magnitude_analysis"])
    args = vars(parser.parse_args())

    create_plots(args)


if __name__ == "__main__":
    main()
