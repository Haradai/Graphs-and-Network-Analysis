import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

# Functions

def retrieve_bidirectional_edges(g: nx.DiGraph, out_filename: str) -> nx.Graph:
    """
    Convert a directed graph into an undirected graph by considering bidirectional edges only.

    :param g: a networkx digraph.
    :param out_filename: name of the file that will be saved.
    :return: a networkx undirected graph.

    """
    undirected_graph = nx.Graph() # Empty undiercted graph

    for u, v in g.edges:
        if g.has_edge(v, u): #Check if it is bidirectional
            undirected_graph.add_edge(u, v)

    nx.write_graphml(undirected_graph, out_filename)

    return undirected_graph

def prune_low_degree_nodes(g: nx.Graph, min_degree: int, out_filename: str) -> nx.Graph:
    """
    Prune a graph by removing nodes with degree < min_degree and removing zero-degree nodes.

    :param g: an undirected networkx graph.
    :param min_degree: lower bound value for the degree.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.
    """
    
    pruned_graph = g.copy()

    # List of nodes with degree < min_degree
    low_degree_nodes = [node for node, degree in pruned_graph.degree if degree < min_degree]
    pruned_graph.remove_nodes_from(low_degree_nodes) # Remove them from the graph

    #Now, we remove the 0 degree
    zero_degree_nodes = [node for node, degree in pruned_graph.degree if degree == 0]
    pruned_graph.remove_nodes_from(zero_degree_nodes)

    # Save the pruned graph to a file
    nx.write_graphml(pruned_graph, out_filename)

    return pruned_graph


def prune_low_weight_edges(g: nx.Graph, min_weight=None, min_percentile=None, out_filename: str = None) -> nx.Graph:
    """
    Prune a graph by removing edges with weight < threshold. Threshold can be specified as a value or as a percentile.

    :param g: a weighted networkx graph.
    :param min_weight: lower bound value for the weight.
    :param min_percentile: lower bound percentile for the weight.
    :param out_filename: name of the file that will be saved.
    :return: a pruned networkx graph.

    """
    if min_weight is not None and min_percentile is not None:
        raise ValueError("Problem detected: Too many arguments. Only one of min_weight or min_percentile should be specified.")
    elif min_weight is None and min_percentile is None:
        raise ValueError("Problem detected: No arguments provided. Either min_weight or min_percentile should be specified.")

    # Graph copy
    pruned_graph = g.copy()

    # Calculate the weight threshold based on min_weight or min_percentile
    if min_weight is not None:
        weight_threshold = min_weight
    else:
        weight_threshold = nx.percentile_weighted(g, min_percentile)

    # Edges with weight < threshold
    low_weight_edges = [(u, v) for u, v, weight in pruned_graph.edges(data="weight") if weight < weight_threshold]
    pruned_graph.remove_edges_from(low_weight_edges)

    # Remove zero-degree nodes
    zero_degree_nodes = [node for node, degree in pruned_graph.degree if degree == 0]
    pruned_graph.remove_nodes_from(zero_degree_nodes)

    # Save the pruned graph to a file if out_filename is specified
    if out_filename is not None:
        nx.write_graphml(pruned_graph, out_filename)

    return pruned_graph

def compute_mean_audio_features(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the mean audio features for tracks of the same artist.

    :param tracks_df: tracks dataframe (with audio features per each track).
    :return: artist dataframe (with mean audio features per each artist).
    """
    # Group tracks by artist and compute the mean of audio features for each artist
    artist_df = tracks_df.groupby('Artist_id').mean()

    # Merge the artist name into the resulting dataframe
    artist_df = artist_df.merge(tracks_df[['Artist_id', 'Artist']], on='Artist_id').drop_duplicates()

    return artist_df


def create_similarity_graph(artist_audio_features_df: pd.DataFrame, similarity: str, out_filename: str = None) -> nx.Graph:
    """
    Create a similarity graph from a dataframe with mean audio features per artist.

    :param artist_audio_features_df: dataframe with mean audio features per artist.
    :param similarity: the name of the similarity metric to use (e.g. "cosine" or "euclidean").
    :param out_filename: name of the file that will be saved.
    :return: a networkx graph with the similarity between artists as edge weights.

    """
    graph = nx.Graph()

    # Compute similarity matrix
    if similarity == 'cosine':
        similarity_matrix = cosine_distances(artist_audio_features_df.drop(['Artist', 'Artist_id'], axis=1)) #We need to drop the two colums that are not numbers!!

    elif similarity == 'euclidean':
        similarity_matrix = euclidean_distances(artist_audio_features_df.drop(['Artist', 'Artist_id'], axis=1))

    else:
        raise ValueError("Invalid similarity metric. Supported metrics: 'cosine', 'euclidean'.")


    for index, row in artist_audio_features_df.iterrows():
        graph.add_node(index, artist_name=row["Artist"])

    num_artists = len(artist_audio_features_df)

    for i in range(num_artists):
        for j in range(i+1, num_artists):
            graph.add_edge(i, j, weight=similarity_matrix[i, j])

    if out_filename is not None:
        nx.write_graphml(graph, out_filename)

    return graph

# Run

if __name__ == "__main__":

    gB = nx.read_graphml("Lab 1/g_gB.graphml")
    gD = nx.read_graphml("Lab 1/g_gD.graphml")

    gB_prime = retrieve_bidirectional_edges(gB, out_filename="Lab 2/g_gB_prime.graphml")
    gD_prime = retrieve_bidirectional_edges(gD, out_filename="Lab 2/g_gD_prime.graphml")

    tracks_df = pd.read_csv('Lab 1/D.csv')

    artists_gB_df = tracks_df[tracks_df["Artist_id"].isin(gB.nodes)]
    artists_gD_df = tracks_df[tracks_df["Artist_id"].isin(gD.nodes)]

    artists_gB_df = compute_mean_audio_features(artists_gB_df)
    artists_gD_df = compute_mean_audio_features(artists_gD_df)

    artists_gB_df.to_csv('Lab 2/artists_gB.csv', index=False)
    artists_gD_df.to_csv('Lab 2/artists_gD.csv', index=False)

    similarity_graph_gB = create_similarity_graph(artist_audio_features_df= artists_gB_df, similarity= "cosine") 
    similarity_graph_gD = create_similarity_graph(artist_audio_features_df= artists_gD_df, similarity= "cosine") 

    #save similarity graph to file
    gB_pruned = prune_low_weight_edges(similarity_graph_gB, min_weight=0, out_filename="Lab 2/gB_pruned.graphml")
    gD_pruned = prune_low_weight_edges(similarity_graph_gD, min_weight=0, out_filename="Lab 2/gD_pruned.graphml")
      
    gB_prime_order = gB_prime.order()
    gB_prime_size = gB_prime.size()

    gD_prime_order = gD_prime.order()
    gD_prime_size = gD_prime.size()

    gB_pruned_order = gB_pruned.order()
    gB_pruned_size = gB_pruned.size()

    gD_pruned_order = gD_pruned.order()
    gD_pruned_size = gD_pruned.size()

    print("g'B:")
    print("Order:", gB_prime_order)
    print("Size:", gB_prime_size)

    print("g'D:")
    print("Order:", gD_prime_order)
    print("Size:", gD_prime_size)

    print("gwB:")
    print("Order:", gB_pruned_order)
    print("Size:", gB_pruned_size)

    print("gwD:")
    print("Order:", gD_pruned_order)
    print("Size:", gD_pruned_size)





