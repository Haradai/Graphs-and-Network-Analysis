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
    artist_audio_features_df = compute_mean_audio_features(tracks_df)

    similarity_graph = create_similarity_graph(artist_audio_features_df= artist_audio_features_df, similarity= "cosine", out_filename= None) 
    g_pruned = prune_low_weight_edges(similarity_graph, min_weight=0.5, out_filename="Lab 2/g_pruned.graphml")

    gB_prime_order = gB_prime.order()
    gB_prime_size = gB_prime.size()

    gD_prime_order = gD_prime.order()
    gD_prime_size = gD_prime.size()


    print("Order and Size of the Undirected Graphs:")
    print("g'B: Order =", gB_prime_order, " Size =", gB_prime_size)
    print("g'D: Order =", gD_prime_order, " Size =", gD_prime_size)





