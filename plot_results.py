import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot Styling Parameters
params = {'legend.fontsize': 'small',
         'axes.labelsize': 'medium',
         'axes.titlesize': 'medium',
         'xtick.labelsize': 'small',
         'ytick.labelsize': 'small'}

# Set Seaborn style and context
sns.set_context(rc=params)
color_palette = sns.color_palette('colorblind')

# Set default marker styles for models
default_marker_styles = {'GCN': 'o', 'Chebyshev GCN': 's', 'GNN': 'D', 'GAT': 'X'}

def stylize_axes(ax, title):
    """
    Stylize the axes by removing the spines and ticks.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to stylize.
    title : str
        The title to set for the axis.
    """
    # removes the top and right lines from the plot rectangle
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_tick_params(top=False, direction='out', width=1)
    ax.yaxis.set_tick_params(right=False, direction='out', width=1)

    # Enforce the size of the title, label, and tick labels
    ax.set_xlabel(ax.get_xlabel(), fontsize='large')
    ax.set_ylabel(ax.get_ylabel(), fontsize='large')

    ax.set_yticklabels(ax.get_yticklabels(), fontsize='medium')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize='medium')

    ax.set_title(title, fontsize='large')

def save_image(fig, title, directory="figs"):
    """
    Save the figure as PNG and PDF files to a specific directory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    title : str
        The name to use for the saved file.
    directory : str
        The directory where to save the file.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Construct full path for the save files
    png_path = os.path.join(directory, title + ".png")
    # pdf_path = os.path.join(directory, title + ".pdf")

    # Save the figure
    fig.savefig(png_path, dpi=300, bbox_inches='tight', transparent=True)
    # fig.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved figures to {png_path}")

def plot_model_accuracy(res_df, graph_sizes, marker_styles=None, filename='accuracy_vs_graph_size'):
    """
    Plots the model accuracy against graph size.

    Parameters
    ----------
    res_df : pd.DataFrame
        The DataFrame containing experiment results.
    graph_sizes : list
        List of graph sizes to use for the x-axis.
    marker_styles : dict, optional
        Dictionary of marker styles for each model.
    filename : str, optional
        The name of the output file (without extension) to save the plot.
    """
    if marker_styles is None:
        marker_styles = default_marker_styles

    plt.figure(figsize=(8, 6))
    plt.xticks(graph_sizes)

    # Plot the data
    sns.lineplot(
        data=res_df,
        x='Graph Size',
        y='Accuracy (%)',
        hue='Model',
        style='Model',
        markers=marker_styles,
        dashes=False,
        palette=color_palette,
        markersize=8,
        linewidth=2,
        ci=None
    )

    # Set plot title and labels
    plt.title('Model Accuracy vs Graph Size', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Graph Size (Number of Nodes)', fontsize=12)

    plt.legend(title='Model', fontsize=10, title_fontsize=12, loc='best')
    plt.tight_layout()

    # Save the plot to "figs" folder
    save_image(plt.gcf(), filename)

    # Display the plot
    plt.show()

def plot_execution_time(res_df, graph_sizes, marker_styles=None, filename='execution_time_vs_graph_size'):
    """
    Plots the model execution time against graph size.

    Parameters
    ----------
    res_df : pd.DataFrame
        The DataFrame containing experiment results.
    graph_sizes : list
        List of graph sizes to use for the x-axis.
    marker_styles : dict, optional
        Dictionary of marker styles for each model.
    filename : str, optional
        The name of the output file (without extension) to save the plot.
    """
    if marker_styles is None:
        marker_styles = default_marker_styles

    plt.figure(figsize=(8, 6))
    plt.xticks(graph_sizes)

    # Plot the data
    sns.lineplot(
        data=res_df,
        x='Graph Size',
        y='Execution Time (s)',
        hue='Model',
        style='Model',
        markers=marker_styles,
        dashes=False,
        palette=color_palette,
        markersize=8,
        linewidth=2,
        ci=None
    )

    # Set plot title and labels
    plt.title('Model Execution Time vs Graph Size', fontsize=14)
    plt.ylabel('Execution Time (s)', fontsize=12)
    plt.xlabel('Graph Size (Number of Nodes)', fontsize=12)

    plt.legend(title='Model', fontsize=10, title_fontsize=12, loc='best')
    plt.tight_layout()

    # Save the plot to "figs" folder
    save_image(plt.gcf(), filename)

    # Display the plot
    plt.show()

if __name__ == "__main__":
    res_df = pd.read_csv("results/full_exp_results.csv")  

    graph_sizes = [10, 30, 50, 100]

    # Plot model accuracy
    plot_model_accuracy(res_df, graph_sizes)

    # Plot execution time
    plot_execution_time(res_df, graph_sizes)
