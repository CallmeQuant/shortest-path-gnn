import numpy as np
import torch
import osmnx as ox
import requests
import time
import random
import networkx as nx
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Tuple, Dict, Any

def download_street_network(place_name: str, network_type: str = 'drive') -> nx.MultiDiGraph:
    """
    Downloads the street network for the specified place and network type.

    Parameters
    ----------
    place_name : str
        The name of the place for which to download the street network.
    network_type : str, optional
        The type of street network to download (e.g., 'drive', 'walk'), by default 'drive'.

    Returns
    -------
    nx.MultiDiGraph
        A NetworkX MultiDiGraph representing the downloaded street network with edge lengths.
    """
    print(f"Downloading street network for {place_name}...")
    G = ox.graph_from_place(place_name, network_type=network_type, simplify=True)
    print("Street network downloaded.")

    if not all('length' in data for _, _, data in G.edges(data=True)):
        print("Adding edge lengths...")
        G = ox.add_edge_lengths(G)
        print("Edge lengths added.")
    else:
        print("All edges have 'length' attribute.")

    return G

def create_robust_session() -> requests.Session:
    """
    Creates a robust session for making HTTP requests with retries, backoff, and a custom User-Agent.

    Returns
    -------
    requests.Session
        A configured Session object with retry strategy.
    """
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    session.headers.update({'User-Agent': 'YourAppName/1.0'})  # Customize as needed
    return session

def get_place_name(lat: float, lon: float, session: requests.Session) -> str:
    """
    Fetches the place name for given coordinates using OpenStreetMap's Nominatim API.

    Parameters
    ----------
    lat : float
        Latitude of the location.
    lon : float
        Longitude of the location.
    session : requests.Session
        A robust session object for making HTTP requests.

    Returns
    -------
    str
        The name of the place corresponding to the provided coordinates, or 'Unknown' if not found.
    """
    url = (
        f"https://nominatim.openstreetmap.org/reverse?"
        f"format=json&lat={lat}&lon={lon}&zoom=18&addressdetails=1"
    )
    try:
        time.sleep(random.uniform(1, 2))  # Respectful delay between requests
        response = session.get(url)
        response.raise_for_status()
        data = response.json()
        return data.get('name', 'Unknown')
    except requests.exceptions.RequestException as e:
        print(f"Error fetching place name for coordinates ({lat}, {lon}): {e}")
        return 'Unknown'

def assign_weights_to_edges(G: nx.MultiDiGraph, edges_csv: str = 'district1_edges.csv') -> nx.MultiDiGraph:
    """
    Ensures all edges have a 'weight' attribute, assigning it based on 'length' or a default value.

    Parameters
    ----------
    G : nx.MultiDiGraph
        The graph whose edges need weight assignment.
    edges_csv : str, optional
        The path to the CSV file containing edge information, by default 'district1_edges.csv'.

    Returns
    -------
    nx.MultiDiGraph
        The updated graph with all edges containing a 'weight' attribute.
    """
    print("Assigning 'weight' to edges...")
    edges_without_weight = [
        (u, v, k) for u, v, k, data in G.edges(keys=True, data=True) if 'weight' not in data
    ]
    print(f"Number of edges without 'weight': {len(edges_without_weight)}")

    for u, v, k in edges_without_weight:
        edge_data = G[u][v][k]
        G[u][v][k]['weight'] = edge_data.get('length', 1)  # Assign 'length' or default to 1

    print("All edges have 'weight' attribute.")
    return G

def check_consistent_node_ids(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Converts node IDs to integers if they are floats representing whole numbers.

    Parameters
    ----------
    G : nx.MultiDiGraph
        The graph whose node IDs need to be checked for consistency.

    Returns
    -------
    nx.MultiDiGraph
        The updated graph with consistent node ID data types.
    """
    print("Checking consistent node ID data types...")
    mapping: Dict[Any, Any] = {}
    for node in G.nodes():
        if isinstance(node, float) and node.is_integer():
            mapping[node] = int(node)

    if mapping:
        G = nx.relabel_nodes(G, mapping, copy=False)
        print(f"Relabeled {len(mapping)} nodes from float to int.")
    else:
        print("No relabeling needed. All node IDs are consistent.")

    return G

def extract_nodes_edges_to_csv(
    G: nx.MultiDiGraph,
    nodes_csv: str = 'district1_nodes.csv',
    edges_csv: str = 'district1_edges.csv'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extracts nodes and edges from the graph, adds place names, and saves them to CSV files.
    Includes one-way road information for edges.

    Parameters
    ----------
    G : nx.MultiDiGraph
        The graph from which to extract nodes and edges.
    nodes_csv : str, optional
        The path to the CSV file where node information will be saved, by default 'district1_nodes.csv'.
    edges_csv : str, optional
        The path to the CSV file where edge information will be saved, by default 'district1_edges.csv'.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing the nodes DataFrame and edges DataFrame.
    """
    print("Extracting nodes and edges...")
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    # Prepare nodes DataFrame
    nodes_df = nodes.reset_index()
    nodes_df.rename(columns={'osmid': 'node_id'}, inplace=True)

    if 'x' not in nodes_df.columns or 'y' not in nodes_df.columns:
        nodes_df['x'] = nodes_df.geometry.x
        nodes_df['y'] = nodes_df.geometry.y

    # Add degree for each node
    nodes_df['degree'] = nodes_df['node_id'].map(dict(G.degree()))

    # Add place names
    print("Fetching place names for nodes (this may take some time)...")
    session = create_robust_session()
    nodes_df['place_name'] = nodes_df.apply(
        lambda row: get_place_name(row['y'], row['x'], session), axis=1
    )
    print("Place names added.")

    # Save nodes to CSV
    nodes_df.to_csv(nodes_csv, index=False)
    print(f"Nodes saved to '{nodes_csv}'.")

    # Prepare edges DataFrame
    edges_df = edges.reset_index()
    edges_df.rename(columns={'u': 'source', 'v': 'target'}, inplace=True)

    # Include one-way information
    edges_df['oneway'] = edges_df['oneway'].fillna(False)  # Default to False if not specified

    # Select relevant columns
    edges_df = edges_df[['source', 'target', 'length', 'oneway']]

    # Save edges to CSV
    edges_df.to_csv(edges_csv, index=False)
    print(f"Edges saved to '{edges_csv}'.")

    return nodes_df, edges_df

def prepare_directed_adjacency_matrix(edges_df: pd.DataFrame) -> Tuple[torch.FloatTensor, Dict]:
    """
    Prepare a square adjacency matrix from the edges DataFrame, accounting for one-way roads.

    Parameters
    ----------
    edges_df : pd.DataFrame
        DataFrame containing edge information with 'source', 'target', and 'oneway' columns.

    Returns
    -------
    torch.FloatTensor
        Adjacency matrix representing the directed edges.
    Dict
        Mapping from node identifiers to their corresponding indices in the adjacency matrix.
    """
    all_nodes = set(edges_df['source']).union(set(edges_df['target']))
    n = len(all_nodes)
    node_to_index = {node: i for i, node in enumerate(all_nodes)}

    adjacency_matrix = np.zeros((n, n))
    for _, row in edges_df.iterrows():
        i, j = node_to_index[row['source']], node_to_index[row['target']]
        adjacency_matrix[i, j] = 1
        if not row['oneway']:
            adjacency_matrix[j, i] = 1

    return torch.FloatTensor(adjacency_matrix), node_to_index

def prepare_weighted_directed_adjacency_matrix(edges_df: pd.DataFrame) -> Tuple[torch.FloatTensor, Dict]:
    """
    Prepare a square adjacency matrix from the edges DataFrame, accounting for one-way roads
    and incorporating edge weights (lengths).

    Parameters
    ----------
    edges_df : pd.DataFrame
        DataFrame containing edge information with 'source', 'target', 'oneway', and 'length' columns.

    Returns
    -------
    torch.FloatTensor
        Weighted adjacency matrix representing the directed edges.
    Dict
        Mapping from node identifiers to their corresponding indices in the adjacency matrix.
    """
    all_nodes = set(edges_df['source']).union(set(edges_df['target']))
    n = len(all_nodes)
    node_to_index = {node: i for i, node in enumerate(all_nodes)}

    adjacency_matrix = np.zeros((n, n))
    for _, row in edges_df.iterrows():
        i, j = node_to_index[row['source']], node_to_index[row['target']]
        adjacency_matrix[i, j] = row['length']
        if not row['oneway']:
            adjacency_matrix[j, i] = row['length']

    return torch.FloatTensor(adjacency_matrix), node_to_index

def prepare_feature_matrix(nodes_df: pd.DataFrame, node_to_index: Dict) -> torch.FloatTensor:
    """
    Prepare the feature matrix using latitude, longitude, and degree.

    Parameters
    ----------
    nodes_df : pd.DataFrame
        DataFrame containing node information with 'node_id', 'y', 'x', and 'degree' columns.
    node_to_index : Dict
        Mapping from node identifiers to their corresponding indices in the feature matrix.

    Returns
    -------
    torch.FloatTensor
        Feature matrix where each row corresponds to a node's [latitude, longitude, degree].
    """
    n = len(node_to_index)
    feature_matrix = np.zeros((n, 3))

    for _, row in nodes_df.iterrows():
        if row['node_id'] in node_to_index:
            i = node_to_index[row['node_id']]
            feature_matrix[i] = [row['y'], row['x'], row['degree']]

    return torch.FloatTensor(feature_matrix)