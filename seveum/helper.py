import json
import networkx as nx
import numpy as np
from typing import Dict, List, Union, Any
from gensim.models import Word2Vec
import random

def serialize_graph(graph: nx.Graph) -> Dict[str, Any]:
    """Serialize NetworkX graph to a JSON-serializable dictionary."""
    return {
        "nodes": [[n, graph.nodes[n]] for n in graph.nodes()],
        "edges": [[u, v, graph[u][v]] for u, v in graph.edges()]
    }

def deserialize_graph(graph_dict: Dict[str, Any]) -> nx.Graph:
    """Deserialize dictionary back to NetworkX graph."""
    G = nx.Graph()
    
    # Add nodes with attributes
    for node, attrs in graph_dict["nodes"]:
        G.add_node(node, **attrs)
    
    # Add edges with attributes
    for u, v, attrs in graph_dict["edges"]:
        G.add_edge(u, v, **attrs)
    
    return G

def generate_graph2vec_embedding(graphs: List[nx.Graph], dimensions: int = 128) -> np.ndarray:
    """Generate embeddings for a list of graphs using node2vec-like approach."""
    def generate_random_walks(G, num_walks=10, walk_length=80):
        walks = []
        nodes = list(G.nodes())
        for _ in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = [str(node)]
                for _ in range(walk_length - 1):
                    curr = walk[-1]
                    neighbors = list(G.neighbors(curr))
                    if neighbors:
                        walk.append(str(random.choice(neighbors)))
                    else:
                        break
                walks.append(walk)
        return walks

    # Generate walks for each graph
    all_walks = []
    for graph in graphs:
        walks = generate_random_walks(graph)
        all_walks.extend(walks)

    # Train Word2Vec model
    model = Word2Vec(sentences=all_walks, vector_size=dimensions, window=5, 
                    min_count=0, sg=1, workers=4)
    
    # Generate graph embeddings by averaging node embeddings
    embeddings = []
    for graph in graphs:
        node_embeddings = [model.wv[str(node)] for node in graph.nodes()]
        if node_embeddings:
            graph_embedding = np.mean(node_embeddings, axis=0)
        else:
            graph_embedding = np.zeros(dimensions)
        embeddings.append(graph_embedding)

    return np.array(embeddings)

def save_graph_to_json(graph: nx.Graph, filepath: str):
    """Save graph to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(serialize_graph(graph), f)

def load_graph_from_json(filepath: str) -> nx.Graph:
    """Load graph from JSON file."""
    with open(filepath, 'r') as f:
        return deserialize_graph(json.load(f))

def save_embeddings_to_json(embeddings: np.ndarray, filepath: str):
    """Save embeddings to JSON file."""
    with open(filepath, 'w') as f:
        json.dump({
            "embeddings": embeddings.tolist(),
            "shape": embeddings.shape
        }, f)

def load_embeddings_from_json(filepath: str) -> np.ndarray:
    """Load embeddings from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        return np.array(data["embeddings"])

def save_graph_with_embedding(graph, embedding, output_path):
    """
    Saves both graph and its embedding in a single JSON file.
    
    Args:
        graph: NetworkX graph object
        embedding: numpy array of graph embedding
        output_path: path to save the combined JSON
    """
    import json
    import networkx as nx
    import numpy as np
    
    combined_data = {
        'graph': nx.node_link_data(graph),  # Convert graph to JSON serializable format
        'embedding': embedding.tolist()      # Convert numpy array to list
    }
    
    with open(output_path, 'w') as f:
        json.dump(combined_data, f)

def load_graph_with_embedding(input_path):
    """
    Loads both graph and its embedding from a single JSON file.
    
    Args:
        input_path: path to the JSON file
    Returns:
        tuple: (NetworkX graph object, numpy array of embedding)
    """
    import json
    import networkx as nx
    import numpy as np
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    graph = nx.node_link_graph(data['graph'])
    embedding = np.array(data['embedding'])
    
    return graph, embedding
