import os
import networkx as nx
from helper import load_graph_with_embedding
import logging

logging.basicConfig(level=logging.INFO)

def main():
    try:
        # Define paths
        working_dir = "./jobs_graphs"
        input_path = os.path.join(working_dir, "graph_with_embedding.json")
        output_path = os.path.join(working_dir, "extracted_graph.graphml")
        
        # Load the combined data
        graph, _ = load_graph_with_embedding(input_path)
        
        # Validate the graph
        logging.info(f"Loaded graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Save as GraphML
        nx.write_graphml(graph, output_path)
        logging.info(f"Successfully saved graph to {output_path}")
        
        # Verify the saved file
        test_load = nx.read_graphml(output_path)
        assert len(test_load.nodes) == len(graph.nodes), "Node count mismatch"
        assert len(test_load.edges) == len(graph.edges), "Edge count mismatch"
        logging.info("Verification successful - graph structure preserved")
        
    except FileNotFoundError:
        logging.error(f"Could not find input file: {input_path}")
    except Exception as e:
        logging.error(f"Error during processing: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()
