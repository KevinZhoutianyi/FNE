import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math

# Patterns for extracting data
data_patterns = {
    "datasize": re.compile(r"num_train_samples=(\d*)"),
    "accuracy": re.compile(r"(Whole Num Accuracy|Whole Number Accuracy|Whole Num Acc)\s+([\d.]+)%"),
    "r2": re.compile(r"R\^2\s+([-+]?\d+\.\d+|[-+]?\d+)"),
    "period": re.compile(r"period_base_list=\[(.*?)\]"),
    "log_file": re.compile(r".*\.log$"),
    "nan": re.compile(r"Loss\s+nan", re.IGNORECASE),
    "model_size": re.compile(r"Actual model size \(total parameters\):\s+([\d\.]+)M"),
    "model_name": re.compile(r"model='([^']+)'"),
}
def extract_logs(base_dir, methods, is_model_size=True):
    """Extract data from log files, considering 'usedigitwisetokenizer_True' in folder names."""
    data = []
    model_name = "Unknown"
    for method in methods:
        method_dir = os.path.join(base_dir, method)
        if os.path.isdir(method_dir):
            for folder in os.listdir(method_dir):
                folder_path = os.path.join(method_dir, folder)
                if os.path.isdir(folder_path):
                    # Determine tokenizer type based on folder name
                    is_digitwise = "usedigitwisetokenizer_true" in folder.lower()
                    add_linear = "addlinear_True" if "addlinear_True" in folder else "addlinear_False"

                    log_files = [
                        os.path.join(folder_path, f)
                        for f in os.listdir(folder_path)
                        if data_patterns["log_file"].match(f)
                    ]
                    if not log_files:
                        continue
                    log_file = log_files[-1]
                    with open(log_file, 'r') as file:
                        entry = {"method": method, "period_base_list": None, "add_linear": add_linear}
                        skip_due_to_nan = False

                        for line in file:
                            if data_patterns["nan"].search(line):
                                skip_due_to_nan = True
                                break
                            for key, pattern in data_patterns.items():
                                if match := pattern.search(line):
                                    if key == "datasize":
                                        entry["datasize"] = int(match.group(1))
                                    elif key == "accuracy":
                                        entry["accuracy"] = float(match.group(2))
                                    elif key == "r2":
                                        entry["r2"] = float(match.group(1))
                                    elif key == "period":
                                        entry["period_base_list"] = match.group(1)
                                    elif key == "model_size" and is_model_size:
                                        entry["model_size"] = float(match.group(1))
                                    elif key == "model_name":
                                        model_name = match.group(1).split('/')[-1]
                                    elif "Train dataset length" in line and "datasize" not in entry:
                                        # Fallback to extract datasize from dataset length
                                        dataset_length_match = re.search(r"Train dataset length:\s+(\d+)", line)
                                        if dataset_length_match:
                                            entry["datasize"] = int(dataset_length_match.group(1))

                        if not skip_due_to_nan:
                            # Add tokenizer suffix only for 'regular' methods
                            if "regular" in method.lower():
                                entry["method"] = f"{method} (Digit-Wise)" if is_digitwise else f"{method} (Subword)"
                            else:
                                entry["method"] = method  # Keep original method for 'fne' and 'xval'
                            data.append(entry)
    return pd.DataFrame(data), model_name


def truncate_float(value, decimals= 6):
    """
    Truncate a floating-point number to a given number of decimal places
    without rounding. E.g. truncate_float(3.3333333,  6) -> 3.3333
    """
    factor = 10.0 ** decimals
    return math.floor(value * factor) / factor


def plot_results(
    data,
    x_key,
    y_keys,
    title,
    x_label,
    y_label,
    log_scale=False,
    highlight_pretrained=None,
    y_lim=None,
    legend_location='best',
    legend_location_accuracy='best',
    simple_version=False
):
    """
    Plot results with special markers for 100% accuracy or R^2 = 1,
    and optionally display the best metric (accuracy, r^2, etc.) in the legend,
    truncating values instead of rounding them.

    :param data: pandas DataFrame containing the data
    :param x_key: column name for the x-axis (e.g. "datasize" or "model_size")
    :param y_keys: dict of column_name -> label for the y-axis 
                   (e.g. {"accuracy": "Accuracy (%)", "r2": "R²"})
    :param title: title of the plot
    :param x_label: x-axis label
    :param y_label: default y-axis label if y_keys doesn't provide a specific one
    :param log_scale: whether to use a log scale on the x-axis
    :param highlight_pretrained: method name to highlight with dotted lines
    :param y_lim: dict specifying y-axis limits for each y_key, e.g. {"r2": [0, 1]}
    :param legend_location: location of the legend for non-accuracy plots
    :param legend_location_accuracy: location of the legend specifically for accuracy plots
    """
    # Adjust font sizes
    if simple_version:
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 12,
            'axes.labelsize': 12,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
        })
        figsize = (4, 4)  # Square dimensions for simple version
    else:
        plt.rcParams.update({
            'font.size': 20,
            'axes.titlesize': 20,
            'axes.labelsize': 20,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20
        })
        figsize = (12, 8)
    # Formatter for log-scale x-axis
    formatter = FuncFormatter(lambda x, _: f'{int(x / 1000)}k' if x >= 1000 else str(int(x)))

    # Assign colors: fne -> red, others -> cycle through non-fne-colors
    color_map = {}
    non_fne_colors = ['darkblue', 'darkgreen', 'darkorange', 'purple', 'saddlebrown']
    non_fne_index = 0
    for method in data["method"].unique():
        if "fne" in method.lower():
            color_map[method] = 'red'
        else:
            color_map[method] = non_fne_colors[non_fne_index % len(non_fne_colors)]
            non_fne_index += 1

    # Plot each y_key on a separate figure
    for y_key, ylabel in y_keys.items():
        plt.figure(figsize=figsize)
        offset_map = {}  # for stacking star markers if multiples share the same x
        lines_and_labels = []  # collect (line_obj, label, best_metric) for legend sorting

        # Make the plot outline 50% transparent
        ax = plt.gca()
        for spine in ax.spines.values():
            spine.set_alpha(0.5)  # Set transparency to 50%

        # Determine offset step for star markers
        if y_lim and y_key in y_lim:
            y_range = y_lim[y_key][1] - y_lim[y_key][0]
        else:
            y_min, y_max = data[y_key].min(), data[y_key].max()
            y_range = y_max - y_min
        offset_step = (y_range / 30.0) if y_range > 0 else 0.0

        # Group by method
        for method, group in data.groupby("method"):
            # This is the method-level "best metric" (max of the entire group).
            # We still store sub-group lines separately, but let's keep track of this if needed.
            method_best_metric = None
            if y_key in group.columns and not group[y_key].isna().all():
                method_best_metric = group[y_key].max()

            # Decide line style based on whether method is highlighted
            linestyle = ':' if method == highlight_pretrained else '-'
            color = color_map[method]

            # Special case: if x_key == "model_size" and method is "Finetuned on..."
            # we draw a horizontal line representing the average (or mean).
            if x_key == "model_size" and method.startswith("Finetuned on"):
                fixed_value = group[y_key].mean()  # or pick a single row, if you prefer
                if method_best_metric is not None:
                    truncated_val = truncate_float(method_best_metric, 6)
                    if y_key == "accuracy":
                        label = f"{method} [Best Acc={truncated_val}%]"
                    elif y_key == "r2":
                        label = f"{method} [Best R²={truncated_val}]"
                    else:
                        label = f"{method} [Best {y_key}={truncated_val}]"
                else:
                    label = f"{method} (Pretrained)"

                # Draw the horizontal line
                line_obj = plt.axhline(
                    y=fixed_value,
                    color=color,
                    linestyle='--'
                )
                # For legend ranking, we'll use the "best_metric" as the sorting key.
                best_metric_for_line = method_best_metric if method_best_metric is not None else float('-inf')
                lines_and_labels.append((line_obj, label, best_metric_for_line))

            else:
                # Otherwise, we group by period_base_list (if present)
                for period, sub_group in group.groupby("period_base_list"):
                    sub_group_best_metric = None
                    if y_key in sub_group.columns and not sub_group[y_key].isna().all():
                        sub_group_best_metric = sub_group[y_key].max()
                    if sub_group_best_metric is None:
                        sub_group_best_metric = float('-inf')

                    # Build label
                    if "fne" in method.lower():
                        method_label = (f"Ours" #(Period: {period})
                                        if period else method.capitalize())
                    else:
                        method_label = method.capitalize()

                    # Include best metric in label, truncated
                    truncated_val = truncate_float(sub_group_best_metric,  6)
                    if y_key == "accuracy":
                        method_label += f" [Best Acc={truncated_val}%]"
                    elif y_key == "r2":
                        method_label += f" [Best R²={truncated_val}]"
                    else:
                        method_label += f" [Best {y_key}={truncated_val}]"

                    sub_group = sub_group.sort_values(x_key)
                    line_list = plt.plot(
                        sub_group[x_key],
                        sub_group[y_key],
                        marker='o',
                        linestyle=linestyle,
                        label=method_label,
                        color=color
                    )

                    # Usually plt.plot() returns a list of lines (1 line per y, but we only have 1)
                    if line_list:
                        line_obj = line_list[0]
                        lines_and_labels.append(
                            (line_obj, method_label, sub_group_best_metric)
                        )

                    # Add star markers if "perfect" value: 100% accuracy or R²=1
                    perfect_marker_added = {"accuracy": False, "r2": False}  # Track if we've added stars to the legend for each metric
                    for i in range(len(sub_group)):
                        x_val = sub_group.iloc[i][x_key]
                        y_val = sub_group.iloc[i][y_key]
                        # Check for "perfect" values
                        if (y_key == "accuracy" and y_val == 100) or (y_key == "r2" and y_val == 1):
                            offset = offset_map.get(x_val, 0)
                            # Draw a small circle at the exact data point
                            plt.plot(x_val, y_val, marker='o', color=color)
                            # Draw a star slightly above (offset) the data point
                            star = plt.plot(
                                x_val,
                                y_val + offset + offset_step,
                                marker='*',
                                color=color,
                                markersize=15,
                                zorder=5
                            )[0]
                            offset_map[x_val] = offset + offset_step

                            # Add to legend if not already done
                            if y_key == "accuracy" and not perfect_marker_added["accuracy"]:
                                lines_and_labels.append((star, "Perfect Accuracy", float('inf')))
                                perfect_marker_added["accuracy"] = True
                            elif y_key == "r2" and not perfect_marker_added["r2"]:
                                lines_and_labels.append((star, "Perfect R²", float('inf')))
                                perfect_marker_added["r2"] = True

        if log_scale:
            plt.gca().xaxis.set_major_formatter(formatter)
            plt.xscale('log')

        plt.xlabel(x_label)
        plt.ylabel(ylabel if ylabel else y_label)
        plt.title(f"{title} - {ylabel}")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        # ---------------------------
        # RANK LEGEND BY BEST METRIC
        # ---------------------------
        # Sort lines_and_labels by best_metric in descending order
        lines_and_labels_sorted = sorted(
            lines_and_labels,
            key=lambda x: x[2],  # x[2] is the best_metric
            reverse=True
        )
        perfect_marker_exists = any(label in ["Perfect Accuracy", "Perfect R²"] for _, label, _ in lines_and_labels_sorted)

        # Move the first legend entry to the end only if a perfect marker exists
        if perfect_marker_exists and lines_and_labels_sorted:
            first_item = lines_and_labels_sorted.pop(0)  # Remove the first item
            lines_and_labels_sorted.append(first_item)   # Append it to the end

        # Extract sorted lines and labels
        sorted_lines = [item[0] for item in lines_and_labels_sorted]
        # Simplify based on keywords
        def simplify_label(label):
            if 'Ours' in label:
                return 'ours'
            elif 'Xval' in label:
                return 'xval'
            elif 'Finetuned' in label:
                return 'finetune'
            elif 'digit-wise' in label:
                return 'digit'
            elif 'subword' in label:
                return 'subword'
            elif 'Perfect' in label:
                return 'perfect'
            else:
                return 'unknown'

        sorted_labels = [item[1] for item in lines_and_labels_sorted]
        if simple_version:
            sorted_labels = [simplify_label(label) for label in sorted_labels]
        if y_key == "accuracy":
            plt.legend(sorted_lines, sorted_labels, loc=legend_location_accuracy, framealpha=0.3)
        else:
            plt.legend(sorted_lines, sorted_labels, loc=legend_location, framealpha=0.3)

        # Apply y-axis limits if specified
        if y_lim and y_key in y_lim:
            plt.ylim(y_lim[y_key])

    plt.show()
def plot_accuracy_difference(base_dir, methods, x_key="datasize", y_key="accuracy", is_model_size=True):
    """Plot the accuracy difference between add_linear_True and add_linear_False."""
    # Extract data for both add_linear_True and add_linear_False
    data, _ = extract_logs(base_dir, methods, is_model_size)

    # Debug: Print columns to verify if 'add_linear' exists
    print("Columns in the extracted data:", data.columns)

    # Filter data for add_linear_True and add_linear_False
    data_true = data[data["add_linear"] == "addlinear_True"]
    data_false = data[data["add_linear"] == "addlinear_False"]

    # Sort data by x_key before merging
    data_true = data_true.sort_values(by=x_key)
    data_false = data_false.sort_values(by=x_key)

    # Merge the data on the x_key (e.g., datasize)
    merged_data = pd.merge(data_true, data_false, on=x_key, suffixes=('_true', '_false'))

    # Calculate the accuracy difference
    merged_data["accuracy_diff"] = merged_data[f"{y_key}_true"] - merged_data[f"{y_key}_false"]

    # Sort merged data by x_key before plotting
    merged_data = merged_data.sort_values(by=x_key)

    # Set font sizes and figure size
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 20,
        'axes.labelsize': 20,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20
    })
    figsize = (12, 8)

    # Plot the accuracy difference
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # Make the plot outline 50% transparent
    for spine in ax.spines.values():
        spine.set_alpha(0.5)

    # Plot the accuracy difference
    plt.plot(
        merged_data[x_key],
        merged_data["accuracy_diff"],
        marker='o',
        linestyle='-',
        color='darkblue',  # Color for the difference line
        label="Accuracy Difference"
    )

    # Set labels and title
    plt.xlabel("Training Data Size" if x_key == "datasize" else "Model Size (Million Parameters)")
    plt.ylabel("Accuracy Difference")
    plt.title("Accuracy difference between with and without add linear layer after FNE")

    # Add grid and legend
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best', framealpha=0.3)

    # Apply log scale if x_key is "datasize"
    if x_key == "datasize":
        formatter = FuncFormatter(lambda x, _: f'{int(x / 1000)}k' if x >= 1000 else str(int(x)))
        plt.gca().xaxis.set_major_formatter(formatter)
        plt.xscale('log')

    plt.show()
def main(
    base_dirs,
    methods,
    x_key,
    y_keys,
    is_model_size=True,
    y_lim=None,
    legend_location='best',
    legend_location_accuracy='best',
    simple_version=False
):
    """Main function to process logs and plot results."""
    combined_data = []
    model_name_for_highlight = None  # track last model_name for highlight

    for base_dir in base_dirs:
        data, model_name = extract_logs(base_dir, methods, is_model_size)
        if "pretrained" in base_dir:
            data["method"] = f"Finetuned on {model_name}"
            model_name_for_highlight = f"Finetuned on {model_name}"
        combined_data.append(data)
    combined_df = pd.concat(combined_data, ignore_index=True).sort_values(by=["method", x_key])

    # Dynamically adjust x_label
    if x_key == "datasize":
        x_label = "Training Data Size (Log Scale)" if not is_model_size else "Training Data Size"
    else:
        x_label = "Model Size (Million Parameters)" if is_model_size else "Training Data Size"

    plot_results(
        combined_df,
        x_key=x_key,
        y_keys=y_keys,
        title="Results",
        x_label=x_label,
        y_label="Metric",
        log_scale=(x_key == "datasize"),  # Use log scale for datasize
        highlight_pretrained=model_name_for_highlight,
        y_lim=y_lim,
        legend_location=legend_location,
        legend_location_accuracy=legend_location_accuracy,
        simple_version=simple_version
    )