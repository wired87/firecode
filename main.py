import os
import ast
import ray
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple

# --- Configuration & Mocks ---

# Define the local directory to scan for Python files (used for demonstration)
TARGET_DIR = r""


def sanitize_path(path: str) -> str:
    """Creates a consistent, sanitized ID from an absolute filepath."""
    # Replace separators with double underscore and dot with _dot_ for consistent IDs
    return path.replace(os.path.sep, '__').replace('.', '_dot_')


def get_component_code(source_code: str, node: ast.AST) -> str:
    """Extracts the source code snippet corresponding to an AST node."""
    try:
        # Use line numbers to extract the code snippet
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            lines = source_code.splitlines()
            start = node.lineno - 1
            end = node.end_lineno
            return '\n'.join(lines[start:end])
        return f"Code for {type(node).__name__}"
    except Exception:
        return f"Code for {type(node).__name__} (Extraction Failed)"


def mock_get_embedding(text: str) -> np.ndarray:
    """Mocks an embedding function, generating a random vector based on text hash."""
    # Generates a pseudo-deterministic 128-dimensional embedding
    hash_val = sum(ord(c) for c in text)
    np.random.seed(hash_val % 100)
    return np.random.rand(128)


def mock_gemini_keyword_request(query: str, node_ids: List[str]) -> List[str]:
    """Mocks a Gemini request to generate keywords for a search query."""
    print(f"  🧠 SIMULATED GEMINI: Generating keywords for query: '{query}'")
    # Generate simple, relevant keywords based on common terms
    keywords = query.lower().split()
    if 'mean' in query: keywords.append('calculate_mean')
    if 'analysis' in query: keywords.append('run_analysis')
    if 'debug' in query: keywords.append('fix_bug')
    return keywords


def mock_gemini_debug_request(component_code: str) -> str:
    """Mocks a Gemini request to analyze and debug a specific code component."""
    # A simple, delayed mock response for demonstration
    return f"✨ Code Review Complete:\n\nOriginal Code Snippet (Lines {component_code.splitlines()[0].split()[2]}-...):\n---\n{component_code}\n---\n\nSuggested Fixes:\n1. Rename the variable 'data' to 'input_data' for clarity.\n2. Add proper type hints to the function signature.\n3. The calculation seems correct, but consider adding a try/except block for numerical stability."


# --- Ray Remote Actor ---

# Decorate the class to turn it into a Ray Remote Actor
@ray.remote
class CodeAnalyzer:
    """A Ray actor that builds a code graph, performs similarity search, and initiates debugging."""

    def __init__(self):
        # Initialize an undirected graph to store the codebase structure
        self.G = nx.Graph()
        # Store embeddings to avoid recalculating them during queries
        self.component_embeddings: Dict[str, np.ndarray] = {}

    # --- Step 1: Graph Generation ---

    def analyze_file(self, filepath: str):
        """Parses a single file, adds file and component nodes, and internal edges."""

        abs_filepath = os.path.abspath(filepath)
        file_id = sanitize_path(abs_filepath)
        print(f"  -> Parsing: {os.path.basename(filepath)} 📂")

        # Read the file content
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Add File Node
        self.G.add_node(file_id, type='file', content=content, abs_path=abs_filepath)

        # Parse the code into an Abstract Syntax Tree (AST)
        tree = ast.parse(content)
        component_map: Dict[str, ast.AST] = {}
        class_nodes: Dict[str, str] = {}  # {method_id: class_id}

        # Walk the AST to identify components (classes, functions, imports)
        for node in ast.walk(tree):
            node_type_name = type(node).__name__

            # Check for high-level components and imports
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)):
                # Create a unique component ID
                component_id = f"{file_id}__{node_type_name}_{node.lineno}"
                component_content = get_component_code(content, node)

                # Add Component Node
                self.G.add_node(component_id, type=node_type_name, content=component_content, parent=file_id)
                component_map[component_id] = node

                # Add Edge: Component contained in File
                self.G.add_edge(component_id, file_id, relationship='contained_in')

                # Handle nested components (methods inside a class)
                if isinstance(node, ast.ClassDef):
                    for sub_node in node.body:
                        if isinstance(sub_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_id = f"{file_id}__{type(sub_node).__name__}_{sub_node.lineno}"
                            class_nodes[method_id] = component_id

        # Add Edge: Child components (methods) to parents (classes)
        for method_id, class_id in class_nodes.items():
            if method_id in self.G.nodes and class_id in self.G.nodes:
                self.G.add_edge(method_id, class_id, relationship='is_part_of')

    def link_dependencies(self):
        """Adds dependency edges (import --> component) based on usage analysis. 🔗"""

        print(f"  -> Linking internal dependencies... 🔗")

        # Iterate over all nodes in the graph
        for node_id in self.G.nodes:
            # Check if the node is an import
            if self.G.nodes[node_id].get('type') in ('Import', 'ImportFrom'):
                import_stmt = self.G.nodes[node_id]['content']
                # Extract the main imported name (e.g., 'numpy' from 'import numpy as np')
                imported_name = import_stmt.split()[1].split('.')[0]
                parent_file_id = self.G.nodes[node_id]['parent']

                # Loop through components in the same file to check for usage
                for neighbor_id in self.G.neighbors(parent_file_id):
                    if self.G.nodes[neighbor_id].get('type') not in ('file', 'Import', 'ImportFrom'):
                        component_content = self.G.nodes[neighbor_id]['content']

                        # Check if the imported name is used in the component's code
                        if imported_name in component_content:
                            # Add Edge: Import uses Component
                            self.G.add_edge(node_id, neighbor_id, relationship='uses_import')

    def build_graph_pipe(self, local_dir: str):
        """Walks the directory, builds the graph, and stores component embeddings. 🏗️"""

        print(f"\n--- 🏗️ STEP 1: Building Graph from {local_dir} ---")

        # Walk local dir
        for root, _, files in os.walk(local_dir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    # Parse each file and add nodes/edges
                    self.analyze_file(filepath)

        # Link dependencies between components
        self.link_dependencies()

        # Generate embeddings for all components and store them
        component_ids = [n for n, attr in self.G.nodes(data=True) if attr['type'] != 'file']

        print("  -> Generating embeddings for components... 🤖")
        for node_id in component_ids:
            # Generate embedding for component content
            node_content = self.G.nodes[node_id]['content']
            self.component_embeddings[node_id] = mock_get_embedding(node_content)

        print(
            f"--- Graph Built: {self.G.number_of_nodes()} nodes, {len(self.component_embeddings)} components embedded. ✅ ---")

        return self.G.number_of_nodes()

    # --- Step 2: Query Pipe ---

    def run_query_pipe(self, text_query: str) -> List[Tuple[str, float]]:
        """Runs similarity search using the graph and returns top 5 component IDs. 🔍"""

        print(f"\n--- 🔍 STEP 2: Running Query Pipe for '{text_query}' ---")

        component_ids = list(self.component_embeddings.keys())
        if not component_ids:
            return []

        # Get keywords from a simulated Gemini request
        keywords = mock_gemini_keyword_request(text_query, component_ids)

        # Combine query and keywords for all query embeddings
        all_query_texts = [text_query] + keywords
        query_embeddings = [mock_get_embedding(text) for text in all_query_texts]

        results: Dict[str, float] = {}

        # Loop through all components
        for component_id, c_embed in self.component_embeddings.items():
            total_score = 0.0
            c_embed_reshaped = c_embed.reshape(1, -1)

            # Perform a similarity search against all query embeddings
            for q_embed in query_embeddings:
                q_embed_reshaped = q_embed.reshape(1, -1)

                # Calculate cosine similarity
                sim_score = cosine_similarity(c_embed_reshaped, q_embed_reshaped)[0][0]
                # Accumulate the scores
                total_score += sim_score

            results[component_id] = total_score

        # Sort the results by score in descending order
        sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)

        # Return top 5 results
        return sorted_results[:5]

    # --- Extras: Debugging ---

    def debug_component(self, component_id: str) -> str:
        """Fetches the component code and sends it to the mock Gemini debugger. 🧠"""

        print(f"\n--- 🧠 STEP 3: Debugging Component: {component_id} ---")

        if component_id not in self.G.nodes:
            return f"Component ID '{component_id}' not found in graph."

        # Retrieve the code content from the graph node
        component_code = self.G.nodes[component_id]['content']

        # Call the mock Gemini debugger to analyze the code
        debug_output = mock_gemini_debug_request(component_code)

        return debug_output


def setup_dummy_project(local_dir: str):
    """Sets up a minimal codebase for the graph analysis to run against."""
    if not os.path.isdir(local_dir):
        os.makedirs(local_dir, exist_ok=True)

    # Create file 1: core logic with an imported dependency
    with open(os.path.join(local_dir, 'core.py'), 'w') as f:
        f.write(
            "# Lines 1-4 are core logic\nimport numpy as np\n\nclass DataProcessor:\n    # Line 6: Class init method\n    def __init__(self, data):\n        self.data = np.array(data)\n\n    # Line 10: Mean calculation method\n    def calculate_mean(self):\n        return self.data.mean() # uses 'mean' and 'data'\n")

    # Create file 2: utility function that imports and uses the class
    with open(os.path.join(local_dir, 'utils.py'), 'w') as f:
        f.write(
            "from core import DataProcessor\n\n# Line 3: Main analysis function\ndef run_analysis(input_list):\n    # Line 5: Instantiate DataProcessor\n    processor = DataProcessor(input_list)\n    # Line 7: Call the method 'calculate_mean'\n    result = processor.calculate_mean()\n    return result\n")

    print(f"Created dummy project structure in '{local_dir}' for testing. 🏗️")


if __name__ == "__main__":

    # Start the Ray environment
    print("🚀 Initializing Ray...")
    # Initialize Ray in local mode for a single-machine test
    ray.init(ignore_reinit_error=True)

    # Setup the dummy project files
    setup_dummy_project(TARGET_DIR)

    # Create an instance of the remote actor
    analyzer_actor = CodeAnalyzer.remote()

    # Run Step 1: Build the graph and embeddings
    # Get the Ray object reference for the result
    build_ref = analyzer_actor.build_graph_pipe.remote(TARGET_DIR)
    # Wait for the graph building to complete
    ray.get(build_ref)

    # Define a user query
    USER_QUERY = "I need to find and fix the function that runs the list analysis and calculates the average."

    # Run Step 2: Query Pipe (Similarity Search)
    print("\n:: next jump in local wf")
    # Execute the query pipe remotely
    query_ref = analyzer_actor.run_query_pipe.remote(USER_QUERY)
    top_results = ray.get(query_ref)

    print("\n--- TOP 5 COMPONENT RESULTS (Query Pipe) ---")

    target_component_id = None

    # Display the top results and save the best match ID for debugging
    for rank, (node_id, score) in enumerate(top_results, 1):
        if rank == 1:
            target_component_id = node_id

        # Retrieve the graph data for better display
        node_data_ref = analyzer_actor.G.get.remote(node_id)
        node_data = ray.get(node_data_ref)

        # Extract a simpler name from the ID
        simple_name = node_id.split('__')[-1].replace('_dot_', '.')
        parent_file = os.path.basename(
            ray.get(analyzer_actor.G.nodes.get.remote(node_data.get('parent', ''))).get('abs_path', 'N/A'))

        print(f"[{rank}] Score: {score:.4f}")
        print(f"    Type: {node_data['type']}, ID: {simple_name}")
        print(f"    File: {parent_file}")

    # Run Step 3: Debug the highest-scoring component
    if target_component_id:
        print("\n:: next jump in local wf")
        # Call the remote debug method on the top result
        debug_ref = analyzer_actor.debug_component.remote(target_component_id)
        debug_output = ray.get(debug_ref)

        print("\n--- 🧠 DEBUGGING OUTPUT (Step 3) ---")
        print(debug_output)

    # Shut down Ray when finished
    ray.shutdown()
    print("\n\nRay shutdown. Workflow complete. 🎉")
