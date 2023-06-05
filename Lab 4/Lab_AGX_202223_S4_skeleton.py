import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import sys
import csv
import numpy as np

sys.path.append('Lab 3')  # Add the directory containing script1.py to the search path

import Lab_AGX_202223_S3_skeleton as lab3

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #
def plot_degree_distribution(degree_dict: dict, normalized: bool = False, loglog: bool = False) -> None:
    """
    Plot degree distribution from dictionary of degree counts.

    :param degree_dict: dictionary of degree counts (keys are degrees, values are occurrences).
    :param normalized: boolean indicating whether to plot absolute counts or probabilities.
    :param loglog: boolean indicating whether to plot in log-log scale.
    
    """
    degrees = list(degree_dict.keys())
    counts = list(degree_dict.values())

    if normalized:
        total_count = sum(counts)
        probabilities = [count / total_count for count in counts]
        counts = probabilities

    plt.figure()
    if loglog:
        plt.loglog(degrees, counts, 'o')
        plt.xlabel('Degree (log scale)')
        plt.ylabel('Count/Probability (log scale)')

    else:
        plt.plot(degrees, counts, 'o')
        plt.xlabel('Degree')
        plt.ylabel('Count/Probability')

    plt.title('Degree Distribution')
    plt.show()


def plot_audio_features(csv_file, artist1_id, artist2_id):
    """
    Plot a figure comparing the mean audio features of two different artists.

    :param csv_file: Path to the CSV file containing the mean audio features of artists.
    :param artist1_id: ID of artist 1.
    :param artist2_id: ID of artist 2.
    :return: None
    """
    with open(csv_file, "r") as file:
        audio_features = list(csv.DictReader(file))

    artists_audio_feat = pd.DataFrame(audio_features)

    artist1_features = artists_audio_feat.loc[artists_audio_feat['Artist_id'] == artist1_id].iloc[:, 1:]
    artist2_features = artists_audio_feat.loc[artists_audio_feat['Artist_id'] == artist2_id].iloc[:, 1:]

    num_features = len(artist1_features.columns)
    feature_labels = artist1_features.columns

    fig, ax = plt.subplots(figsize=(10, 6))
    index = np.arange(num_features)
    bar_width = 0.35

    # Plotting artist 1
    rects1 = ax.bar(index, artist1_features.values.flatten(), bar_width, label=artist1_id,  color='pink')

    # Plotting artist 2
    rects2 = ax.bar(index + bar_width, artist2_features.values.flatten(), bar_width, label=artist2_id, alpha=0.5,  color='blue')

    ax.set_xlabel('Audio Feature')
    ax.set_ylabel('Mean Value')
    ax.set_title('Comparison of Audio Features')
    ax.set_xticks(index + bar_width/2)
    ax.set_xticklabels(feature_labels, rotation='vertical')
    ax.legend()

    plt.tight_layout()
    plt.show()

import pandas as pd
import plotly.express as px

def plot_similarity_heatmap(csv_file, similarity: str, out_filename: str = None) -> None:
    """
    Plot a heatmap of the similarity between artists.

    :param csv_file: Path to the CSV file containing the mean audio features of artists.
    :param similarity: String with similarity measure to use.
    :param out_filename: Name of the file to save the plot. If None, the plot is not saved.
    """
    # Load data from CSV
    artist_audio_features_df = pd.read_csv(csv_file)

    # Drop NaN values and unnecessary columns
    artist_audio_features_df = artist_audio_features_df.dropna()
    artist_audio_features_df = artist_audio_features_df.drop(['Artist', 'Artist_id'], axis=1)

    # Compute similarity matrix
    similarity_matrix = artist_audio_features_df.iloc[:, 1:].corr(method=similarity)

    # Plot heatmap using Plotly
    fig = px.imshow(similarity_matrix.values,
                    x=artist_audio_features_df.columns[1:],
                    y=artist_audio_features_df.columns[1:],
                    labels=dict(color="Similarity"),
                    color_continuous_scale='RdBu')

    fig.update_layout(
        title=f"Artist Similarity Heatmap ({similarity})",
        xaxis_title="Artists",
        yaxis_title="Artists"
    )

    fig.show()


if __name__ == "__main__":
    
    gB_prime = nx.read_graphml("Lab 2/g_gB_prime.graphml")
    gD_prime = nx.read_graphml("Lab 2/g_gD_prime.graphml")

    dd_gB_prime = lab3.get_degree_distribution(gB_prime)
    dd_gD_prime = lab3.get_degree_distribution(gD_prime)

    plot_degree_distribution(dd_gB_prime, normalized=False, loglog=False)
    plot_degree_distribution(dd_gD_prime, normalized=False, loglog=False)

    #! Not woriking well...
    #TODO: Find most similar artist
    plot_audio_features("Lab 2/mean_audio_features.csv", artist1_id = "3TVXtAsR1Inumwj472S9r4", artist2_id= "1anyVhU62p31KFi8MEzkbf")

    #TODO: Find least similar artist
    plot_audio_features("Lab 2/mean_audio_features.csv", artist1_id = "3TVXtAsR1Inumwj472S9r4", artist2_id= "3vDUJHQtqT3jFRZ2ECXDTi")
    
    #TODO: Also needs to be fixed
    plot_similarity_heatmap("Lab 2/mean_audio_features.csv", similarity="kendall", out_filename="heatmap.png")
