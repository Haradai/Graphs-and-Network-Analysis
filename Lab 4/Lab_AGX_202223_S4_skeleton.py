import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import sys
import csv
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

sys.path.append('Lab 3')  # Add the directory containing script1.py to the search path

import Lab_AGX_202223_S3_skeleton as lab3

# ------- IMPLEMENT HERE ANY AUXILIARY FUNCTIONS NEEDED ------- #


# --------------- END OF AUXILIARY FUNCTIONS ------------------ #
import matplotlib.pyplot as plt

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

    name1 = artist1_features["Artist"].values[0]
    name2 = artist2_features["Artist"].values[0]

    artist1_features = artist1_features.drop(["Artist", "Artist_id","Loudness"], axis=1)
    artist2_features = artist2_features.drop(["Artist", "Artist_id","Loudness"], axis=1)

    # Convert the values to numeric
    artist1_features = artist1_features.apply(pd.to_numeric)
    artist2_features = artist2_features.apply(pd.to_numeric)
    
    print(artist2_features)
    print(artist1_features)

    index = np.arange(10)
    bar_width = 0.35

    fig, ax = plt.subplots()
    summer = ax.bar(index, np.abs(np.log2(artist1_features.values[0])), bar_width, label=name1)

    winter = ax.bar(index+bar_width, np.abs(np.log2(artist2_features.values[0])), bar_width, label=name2)

    ax.set_xlabel('Category')
    ax.set_title('Comparing artists')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(['Duration', 'Popularity', 'Danceability', 'Energy',
                'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness',
                'Valence', 'Tempo'], rotation = 90)
    ax.legend()
    plt.tight_layout()
    plt.savefig("plot.png")
   

def plot_similarity_heatmap(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> None:
    """
    Plot a heatmap of the similarity between artists.
    
    :param artist_audio_features_df: dataframe with mean audio features of artists.
    :param similarity: string with similarity measure to use.
    :param out_filename: name of the file to save the plot. If None, the plot is not saved.
    """
    
    features = artist_audio_features_df.drop(columns=['Artist_id', 'Artist'])
    similarity_matrix = cosine_similarity(features) if similarity == 'cosine' else features.corr(method=similarity)
    
    # Plot the heatmap
    sns.set(font_scale=1.2)
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=False, cmap='coolwarm', cbar=True, xticklabels=artist_audio_features_df['Artist'], yticklabels=artist_audio_features_df['Artist'])
    plt.title(f'Similarity Heatmap ({similarity.capitalize()})')
    plt.xlabel('Artists')
    plt.ylabel('Artists')
    plt.xticks(rotation=90, ha='right', fontsize=0.5)
    plt.yticks(rotation=0, fontsize=0.5)
    
    # Save or display the plot
    if out_filename:
        plt.savefig(out_filename, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    
    gB_prime = nx.read_graphml("Lab 2/g_gB_prime.graphml")
    gD_prime = nx.read_graphml("Lab 2/g_gD_prime.graphml")

    dd_gB_prime = lab3.get_degree_distribution(gB_prime)
    dd_gD_prime = lab3.get_degree_distribution(gD_prime)

    #Question 1

    plot_degree_distribution(dd_gB_prime, normalized=False, loglog=True)
    plot_degree_distribution(dd_gD_prime, normalized=False, loglog=True)
    
   #   Question 2 and 3
    gB = nx.read_graphml("Lab 1/g_gB.graphml")

    df = pd.read_csv("Lab 2/mean_audio_features.csv")
    gB_artists = set(gB.nodes())
    df_gB = df[df['Artist_id'].isin(gB_artists)]

    # Extract the audio features of Drake
    drake_features = df_gB[df_gB['Artist_id'] == '3TVXtAsR1Inumwj472S9r4'].iloc[:, 2:-1].values

    # Calculate the cosine similarity between Drake and other artists
    similarities = cosine_similarity(drake_features, df_gB.iloc[:, 2:-1].values)

    # Get the index of the most similar artist
    second_most_similar_index = np.argsort(similarities, axis=None)[-2]
    less_similar_index = np.argsort(similarities, axis=None)[0]
   
    # Retrieve the information of the second most similar artist
    second_most_similar_artist = df_gB.iloc[second_most_similar_index]['Artist']
    less_similar_artist = df_gB.iloc[less_similar_index]['Artist']

    print("The most similar artist to Drake is:", second_most_similar_artist)
    print("The less similar artist to Drake is:", less_similar_artist)

    # Most similar
    plot_audio_features("Lab 2/mean_audio_features.csv", artist1_id = "3TVXtAsR1Inumwj472S9r4", artist2_id= "1RyvyyTE3xzB2ZywiAwp0i")

    # Least siilar
    plot_audio_features("Lab 2/mean_audio_features.csv", artist1_id = "3TVXtAsR1Inumwj472S9r4", artist2_id= "1iJdyDcY98X3GMnUesl7tf")
   
    artist_audio_features_df = pd.read_csv("Lab 2/mean_audio_features.csv")
    artist_audio_features_df = artist_audio_features_df[:200] 

    # Question 4
    plot_similarity_heatmap(artist_audio_features_df, similarity="cosine", out_filename="heatmap.png")

    