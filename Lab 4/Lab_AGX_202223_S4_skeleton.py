import pandas as pd
import matplotlib.pyplot as plt

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

def plot_audio_features(artists_audio_feat: pd.DataFrame, artist1_id: str, artist2_id: str) -> None:
    """
    Plot a (single) figure with a plot of mean audio features of two different artists.

    :param artists_audio_feat: dataframe with mean audio features of artists.
    :param artist1_id: string with id of artist 1.
    :param artist2_id: string with id of artist 2.
    :return: None
    """
    artist1_features = artists_audio_feat.loc[artists_audio_feat['artist_id'] == artist1_id].iloc[:, 1:] #Take audio features from artitst
    artist2_features = artists_audio_feat.loc[artists_audio_feat['artist_id'] == artist2_id].iloc[:, 1:]

    num_features = len(artist1_features.columns) #should be revised
    feature_labels = artist1_features.columns

    fig, ax = plt.subplots(figsize=(10, 6))
    index = range(num_features)
    bar_width = 0.35

    # Plotting artist 1
    rects1 = ax.bar(index, artist1_features.values.flatten(), bar_width, label=artist1_id)

    # Plotting artist 2
    rects2 = ax.bar(index + bar_width, artist2_features.values.flatten(), bar_width, label=artist2_id)

    ax.set_xlabel('Audio Feature')
    ax.set_ylabel('Mean Value')
    ax.set_title('Comparison of Audio Features')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(feature_labels)
    ax.legend()

    plt.tight_layout()
    plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

def plot_similarity_heatmap(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> None:
    """
    Plot a heatmap of the similarity between artists.

    :param artist_audio_features_df: dataframe with mean audio features of artists.
    :param similarity: string with similarity measure to use.
    :param out_filename: name of the file to save the plot. If None, the plot is not saved.
    """
    # Compute similarity matrix
    similarity_matrix = artist_audio_features_df.iloc[:, 1:].corr(method=similarity)

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, cmap='coolwarm', annot=True, fmt=".2f", square=True)

    plt.title(f"Artist Similarity Heatmap ({similarity})")
    plt.xlabel("Artists")
    plt.ylabel("Artists")

    if out_filename:
        plt.savefig(out_filename)

    plt.show()


if __name__ == "__main__":
    # ------- IMPLEMENT HERE THE MAIN FOR THIS SESSION ------- #
    pass
    # ------------------- END OF MAIN ------------------------ #
