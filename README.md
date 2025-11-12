# firecoder

synthax: 
"::" = next jump in local wf
"->" = next pathway step
"-->" = edge connect 
">" bring workflows (wf) = together
"- receive code str ->" = rcs
"receive list prompt" = rlp


legend
similairity search = ss
vectro store = vs

ROLE:
SENIOR SCIENTIFIC SOFTWARE ARCHITECT with the task to build software systems


PROMPT:
Improve the prmpt qualiy followign the 
**CRITERIA:**
Clarity and Specificity 💡
Define the Role (Persona)

Use Direct Imperatives

Specify the Format

Constraint and Scope 📏
Define Constraints (The "Don't" List)

Set the Length and Tone

Provide Examples (Few-Shot Prompting)

Context and Iteration 🖼️
Clarify Necessary Knowledge (Context)

Use Chain-of-Thought (CoT)



GOAL:
Provide an entire codebase genrated with your imroved prompt withing a single file



CONTEXT:

# Relay remote
- receive query string -> create 5 versions of prompt using a local llm (downloadable with pip) > classifier rlp
- qa handling for actions that has been performed. add here qa functionaity in harmony to the terminal input field (render)
- 

# Classifier
rlp -> collect all nodes with type=MODULE ->  perform ss embed_description to 


# Expert 
- - define local vs
  - define static search engine prompt: gather information based specific isntructions to support the underlying goal
  - fetch 
- rlp -> convert to 5 search queries -> gsearch request -> embed and save page content first 5 pages (non ads) in table format within local vs. -> extract: pip packages required for this task -> run subprocess -> load packages in Graph (nid=package_name, ref=package_instance)


# ADD_CODE_GRAPH 
 use ast: identify all present datatypes in each file -> extract and classify file content -> add_node for each with the parent=[parent file-node] and type=datatype(class, def, comment) ´, embed datatype (class/method etc) (only node keys wrapped inside dict(embedding(364dim), nid, t, type), code make available
- collect all packages specified in r.txt -> add_node(dict(nid, package instance, embed description) 
- scan each datatype for sub modules/classes/functions/motheds/packages used from outer space -> link logical to destination(e.g. method A defines a class inside from external) -> add_edge with rel="uses" - classify directions (seppare between imports or imported) -> nx.link_node_data & pyvis render -> save files (json & html) in root -> print_status_G


# Graph Engine
- - create nx.MiltiDiGraph
- walk local dir (exclude .venv) -> add_node all folders type = MODULE, description=embed(llm file request: sum content within  -> extract content each file add_node file_name -> add_edge file_name --> module rel=has_file, ADD_CODE_GRAPH


# Collector remote
INGEST list query (rlp).
SEARCH & SCORE: ITERATE through ALL graph nodes. EMBED node ID (nid) locally and PERFORM Similarity Search (ss) against the query embedding. STORE results in dict(nid: score).
FILTER & PATHFIND: ITERATE through nids where score > 0.9.
EXECUTE bidirectional pathfinding algorithm (get_neighbors rel="needs") starting from each high-score nid (e.g., class/method). AGGREGATE all dependency nodes (including sub-modules) into self.pathway_nodes.
DATA RESOLUTION: EXTRACT required variable data for collected components (method/class headers) directly from the .env file (Ensure missing variables are flagged).
ASSEMBLE & RETURN: TOPOLOGICALLY SORT and COMBINE all retrieved code components and resolved variables into a single, runnable code string. RETURN the code string.


FUNCITONALITIES:
# Executor remote
- rcs -> create runnable end executes the sorted codebase to avoid any issues(like import error) inside a ray.remote -> ADD_CODE_GRAPH(with adapted code)

# editor
- rcs -> llm call gem cli py client: static prompt: perform change on files -> ADD_CODE_GRAPH(generated code content)


# creator 
- rlp -> gem api call: create code base -> ADD_CODE_GRAPH


# ui
**terminal based ui to interact with the engine:**
- whil loop and classifys all query inputs
- blocks main thread
- incldue state management, edit debug isntructions(add, del),  issue / submit handling and possibility to query the engine with relay as first contact after query input
- welcome message
- render possible options numbered 
- answer / follow up question handling
- direct entry point to Relay 

  
# Debugger remote
- while loop runnable current code content entire Graph -> execute in subprocess each workflow -> check traceback and debug wih a gem instance within a while loop (include global debbug instructions in each iter)

run the extend and inmproved prompt to ensure functionality


# extras:
- main call altiems starts the cli. - user controlls the run of specific functinality based on its quwery input 
- use clear oneliner comments before each fuction/method call and at the start of each method to intepret
- use creative prints with emojicons
- check avoid errors
- include the entire setup to init and run ray 
- include a r.txt (requiremens)
- proviede working ready2use code
- define the entire codebase inside a single file
- each worker must be deployed from 
- define a clear step by step workflow hat executes each possibel cli action wrapped isnide a testing def and if name amin call
- include funcitnoality to load picked datatypes(fucntions & classes) for all defined workflows inisde a ray.remote
- define each defined workflow inside a ray.remote
- use the coding schema of all available files to ensure harmony uniformness
- include loop





CODE TO ADAPT:
import ast
import re
import networkx as nx
import json
import os
import glob
from typing import Union, Set, Optional, Dict, Any, List
import jax.numpy as jnp
import numpy as np
import ray
from ray.actor import ActorHandle
from ray.types import ObjectRef
import subprocess  # For r.txt creation in testing
import time

# --- Configuration & Setup ---

# Initialize Ray if not already running (required for remote execution)
if not ray.is_initialized():
    # Use local mode for simplicity in this example
    ray.init(ignore_reinit_error=True)

# --- Constants & Global Types ---
CODEBASE_ROOT = r"C:\Users\bestb\Desktop\qfs"
EXCLUDE_DIRS = {".venv", "__pycache__", ".git"}
ENV_FILE_PATH = "project.env"
REQUIREMENTS_FILE = "r.txt"
OUTPUT_GRAPH_JSON = "code_graph.json"

# NOTE: LLM/Embedding simulation parameters
LLM_EMBEDDING_DIM = 10
SIMILARITY_THRESHOLD = 0.9

BUILTIN_TYPES: Set[str] = {
    'str', 'int', 'float', 'bool', 'list', 'tuple', 'dict', 'set', 'Any',
    'jnp.ndarray', 'complex', 'None', 'ActorHandle', 'ObjectRef', 'bytes'
}


# --- Utility Functions ---

def _get_docstring(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> str:
    """// Extracts docstring from function or class node."""
    return ast.get_docstring(node) or ""


def _get_type_name(node: Optional[ast.expr]) -> str:
    """// Extracts type name from annotation or defaults to 'Any'."""
    return ast.unparse(node) if node else 'Any'


def load_env_variables(path: str) -> Dict[str, str]:
    """// Loads environment variables from the specified path (project.env)."""
    # NOTE: Simplified loading for testing
    if os.path.exists(path):
        # Implement actual file parsing here
        return {"T": "1.0", "MASS": "0.1", "G_COUPLING": "0.6"}
    return {}


def load_requirements(path: str) -> List[str]:
    """// Loads package names from r.txt."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return [line.strip().split('==')[0] for line in f if line.strip() and not line.startswith('#')]
    return []


# --- Embedding Simulation ---
def embed_text_sim(text: str) -> np.ndarray:
    """
    // Def: create 5 versions of prompt using a local llm downloadable with pip -> embed them
    // Simuliert die Erstellung eines Embeddings für einen Text (Node Code/Prompt).
    """
    # Simuliert das 10-dimensionale Embedding-Array
    random_seed = sum(ord(c) for c in text) % 1000
    np.random.seed(random_seed)
    return np.random.rand(LLM_EMBEDDING_DIM)


def get_similarity_score_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """// Berechnet die Ähnlichkeit (Cosinus-Ähnlichkeit) zwischen zwei Embeddings (Simuliert)."""
    # Cosine Similarity = dot_product(A, B) / (norm(A) * norm(B))
    dot_product = np.dot(emb1, emb2)
    norm_a = np.linalg.norm(emb1)
    norm_b = np.linalg.norm(emb2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.clip(dot_product / (norm_a * norm_b), 0.0, 1.0)


# --- Graph & Serialisierung Helpers ---

class Manipulator:
    """// Helper class to encapsulate cleaning functions."""

    def replace_special_chars(self, s: Union[str, bytes]) -> str:
        """// Sanitizes keys by keeping only alphanumeric and underscore characters."""
        if not isinstance(s, (str, bytes)):
            return ""
        return re.sub(r'[^a-zA-Z0-9_]', '', s)

    def clean_attr_keys(self, attrs: Dict[Any, Any]) -> Dict[str, Any]:
        """// Cleans the keys of a dictionary without modifying the values."""
        cleaned_attrs = {}
        for k, v in attrs.items():
            cleaned_key = self.replace_special_chars(k)
            cleaned_attrs[cleaned_key] = v
        return cleaned_attrs


class LocalGraphUtils:
    """// Graph utility class using NetworkX MultiDiGraph."""

    def __init__(self):
        # 🚀 create nx.MiltiDiGraph
        self.G = nx.MultiDiGraph()
        self.manipulator = Manipulator()
        self.pathway_nodes = set()
        print("🧭 Graph Initialized (MultiDiGraph)")

    def add_node(self, attrs: Dict[str, Any]):
        """// Adds a node with sanitized attributes."""
        node_id = attrs["nid"]
        # NOTE: embed datatype (class/method etc) (only node keys wrapped inside dict(embedding, nid, t, type, code)
        # 💡 Generiere Embedding für Code-Inhalte, falls vorhanden
        code_content = attrs.get('code', attrs.get('nid', ''))
        attrs['embedding'] = embed_text_sim(code_content).tolist()  # Speichere als Liste für JSON

        cleaned_attrs = self.manipulator.clean_attr_keys(attrs)
        self.G.add_node(node_id, **cleaned_attrs)

    def add_edge(self, src: str, trgt: str, attrs: Dict[str, Any]):
        """// Adds a directed edge with sanitized attributes."""
        cleaned_attrs = self.manipulator.clean_attr_keys(attrs)
        self.G.add_edge(src, trgt, **cleaned_attrs)

    def get_node(self, nid: str) -> Dict[str, Any]:
        """// Retrieves a node's attributes."""
        return self.G.nodes.get(nid, {})

    # 🗺️ Pathfinding Logic (AAA)
    def get_neighbors_recursive(self, start_node: str, relationship: str = "uses") -> Set[str]:
        """// Pathfinding algorithm (BFS) to find dependencies (AAA)."""
        self.pathway_nodes = set()
        queue = [start_node]

        while queue:
            current_node = queue.pop(0)
            if current_node in self.pathway_nodes:
                continue

            self.pathway_nodes.add(current_node)

            # 💡 rel="needs" ist der logische Abstraktionsschritt von rel="uses"
            # Hier wird 'uses' und 'needs_input' gefolgt
            for _, neighbor, edge_attrs in self.G.out_edges(current_node, data=True):
                relationship_type = edge_attrs.get('rel')

                # Sammle Nachbarn (collect neighbor nodes global in class attr "self.pathway_nodes")
                if relationship_type in [relationship, 'needs_input', 'has_method', 'has_class']:
                    # next jump in local wf :: enqueue neighbor
                    queue.append(neighbor)

        return self.pathway_nodes


class StructInspector(ast.NodeVisitor):
    """// Traverses AST to map code structure and dependencies."""

    def __init__(self, lex: LocalGraphUtils, module_name: str, file_content: str):
        self.current_class: Optional[str] = None
        self.lex = lex
        self.module_name = module_name
        self.file_content = file_content
        # Add root file node
        self.lex.add_node({"nid": module_name, "type": "FILE_MODULE", "code": file_content})
        self.imported_components = {}

    def visit_ClassDef(self, node: ast.ClassDef):
        # 1. Add CLASS Node
        self.current_class = node.name
        self.lex.add_node({"nid": node.name, "parent": [self.module_name], "type": "CLASS", "code": ast.unparse(node)})
        # 2. Link MODULE -> CLASS --> edge connect
        self.lex.add_edge(src=self.module_name, trgt=node.name, attrs={'rel': 'has_class'})
        self.generic_visit(node)
        self.current_class = None

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        """// Processes methods/functions, parameters, and return types."""
        method_id = node.name
        return_key = self.extract_return_statement_expression(node)
        entire_def = ast.unparse(node)
        parent_id = self.current_class or self.module_name

        # 1. Add METHOD Node
        self.lex.add_node({
            "nid": method_id,
            "parent": [parent_id],
            "type": "METHOD",
            'return_key': return_key,
            "code": entire_def,
        })

        # 2. Link PARENT -> METHOD --> edge connect
        self.lex.add_edge(src=parent_id, trgt=method_id, attrs=dict(rel='has_method'))

        # 3. Process Parameters (Method header)
        for arg in node.args.args:
            if arg.arg == 'self': continue
            param_name = arg.arg
            param_type = _get_type_name(arg.annotation)

            # Add PARAM Node
            self.lex.add_node({"nid": param_name, "type": "PARAM", "parent": [method_id]})

            # Link METHOD -> PARAM rel="needs"
            self.lex.add_edge(
                src=method_id, trgt=param_name,
                attrs=dict(rel='needs', type=param_type))  # rel="needs" is crucial for pathfinding

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """// Scans for functions/methods used from outer space."""
        # Find called name
        called_name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else None)
        if called_name:
            current_component = self.current_class or self.module_name

            # Link current component -> used function/method
            self.lex.add_edge(
                src=current_component,
                trgt=called_name,
                attrs={'rel': 'uses', 'direction': 'out'}
            )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        # 🚀 Add_edge with rel="uses" - classify directions (imports)
        for alias in node.names:
            self.lex.add_node({"nid": alias.name, "type": "PACKAGE_REF"})
            self.lex.add_edge(
                src=self.module_name, trgt=alias.name,
                attrs={'rel': 'uses', 'direction': 'import'}
            )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        # 🚀 Add_edge with rel="uses" - classify directions (imports)
        source = node.module
        for alias in node.names:
            self.lex.add_edge(
                src=self.module_name, trgt=alias.name,
                attrs={'rel': 'uses', 'source_module': source}
            )
        self.generic_visit(node)

    def extract_return_statement_expression(self, method_node: ast.FunctionDef) -> Optional[str]:
        """// Extracts the source code expression being returned."""
        for node in ast.walk(method_node):
            if isinstance(node, ast.Return) and node.value is not None:
                return ast.unparse(node.value).strip()
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node)

    def visit_Name(self, node: ast.Name):
        self.generic_visit(node)


# --- Workflow Implementation ---

class CodeAnalyzerCLI:
    """// Manages the full workflow and executes CLI actions via Ray."""

    def __init__(self, codebase_root=CODEBASE_ROOT):
        # Final graph storage and utilities
        self.code_graph = LocalGraphUtils()
        self.codebase_root = codebase_root
        self.env_vars = load_env_variables(ENV_FILE_PATH)
        self.required_packages = load_requirements(REQUIREMENTS_FILE)
        self.pathway_nodes: Set[str] = set()  # For pathfinding results
        print("💡 CodeAnalyzerCLI Initialized.")

    # --- Step 1: Graph Construction ---
    def create_nx_graph(self):
        """// Workflow Step 1: Builds the graph structure from the codebase."""
        print("\n=== 1. START: Graph Construction ===")

        # 🚀 walk local dir (exclude .venv)
        for root, dirs, files in os.walk(self.codebase_root):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for file_name in files:
                if file_name.endswith(".py"):
                    file_path = os.path.join(root, file_name)
                    module_name = file_name.replace(".py", "")

                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()

                    # 🛠️ Use AST to extract and classify content
                    try:
                        inspector = StructInspector(self.code_graph, module_name, content)
                        tree = ast.parse(content)
                        inspector.visit(tree)
                    except Exception as e:
                        print(f"⚠️ Could not parse {file_name}: {e}")

        # 🚀 collect all packages specified in r.txt -> add_node
        for package in self.required_packages:
            self.code_graph.add_node({
                "nid": package,
                "type": "PACKAGE",
                "parent": ["DEPENDENCIES"],
                "description": f"External dependency from {REQUIREMENTS_FILE}",
            })

        print(f"✅ Graph Construction Finished. Total Nodes: {self.code_graph.G.number_of_nodes()}")

    # --- Step 2: Query Pipe and Remote Execution ---

    # 📢 CLI Action 1: Create JSON
    def cli_action_create_json(self, output_path=OUTPUT_GRAPH_JSON) -> str:
        """// @ cli start: create json file from created graph."""
        print(f"\n:: CREATE JSON -> {output_path}")
        try:
            graph_data = nx.node_link_data(self.code_graph.G)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2)
            print(f"✅ JSON file created.")
            return output_path
        except Exception as e:
            print(f"❌ Error serializing graph: {e}")
            return "ERROR"

    # 🗺️ CLI Action 2: Semantic Search (Simulated) and Pathfinding
    def cli_action_sem_search_and_path(self, query: str) -> List[str]:
        """
        // Define workflow: receive string -> create 5 versions of prompt (sim) -> embed ->
        // loop all nodes of the graph embed nid local perform similarity-serach(ss) -> loop nids with sc > .9: pathfinding
        """
        print(f"\n:: SEMANTIC SEARCH & PATHFINDING for query: '{query}'")

        # 1. Embed Query (Simulated)
        query_embedding = embed_text_sim(query)

        high_similarity_nodes: Dict[str, float] = {}

        # 2. Loop all nodes, perform similarity-search(ss)
        for nid, data in self.code_graph.G.nodes(data=True):
            if 'embedding' in data:
                node_embedding = np.array(data['embedding'])
                score = get_similarity_score_sim(query_embedding, node_embedding)

                # loop nids with sc > .9:
                if score >= SIMILARITY_THRESHOLD:
                    high_similarity_nodes[nid] = score

        print(f"🔎 Found {len(high_similarity_nodes)} nodes with similarity > {SIMILARITY_THRESHOLD}")

        final_pathway_nodes = set()

        # 3. Pathfinding on high similarity nodes
        for nid, score in high_similarity_nodes.items():
            print(f"  -> Pathfinding on matching node: {nid} (Score: {score:.3f})")
            # 💡 include a pathfinding algirithmus which receives a nid of a specific datatype
            # We use 'needs' here for the core dependency flow
            path_nodes = self.code_graph.get_neighbors_recursive(nid, relationship="needs")
            final_pathway_nodes.update(path_nodes)

        print(f"✅ Final set of related and dependency nodes: {len(final_pathway_nodes)}")
        return list(final_pathway_nodes)

    # 🚀 CLI Action 3: Remote Execution (Wrapped in Ray Remote)
    @ray.remote
    def _execute_code_remote(self, function_code: str, params: Dict[str, Any]):
        """// Creates and loads entire combined str code base executable in ray.remote -> run function."""
        # This function resides in the Ray remote environment

        # 1. Collect parameters for all params used(method/class header) by specific datatype
        # (Simplified: uses loaded env vars and passed input params)
        final_params = self.env_vars.copy()
        final_params.update(params)

        # 2. Create the executable environment
        exec_globals = {}
        import jax;
        exec_globals['jnp'] = jax.numpy
        exec_globals['jax'] = jax

        # 3. Compile and execute the function code
        try:
            compiled_code = compile(function_code, '<string>', 'exec')
            exec_locals = {}
            exec(compiled_code, exec_globals, exec_locals)

            func_name = list(exec_locals.keys())[-1]
            target_func = exec_locals[func_name]

            # Simple simulation of result
            result = f"Simulated execution of {func_name} complete with params: {final_params}"
            return {"result": result, "function_name": func_name}

        except Exception as e:
            return {"Error": str(e), "Traceback": "See remote log"}

    def cli_action_run_remote(self, node_id: str, input_params: Dict[str, Any] = {}):
        """// CLI Action: Runs a specific METHOD/CLASS remotely."""
        print(f"\n:: REMOTE EXECUTION -> Running code for Node ID: {node_id}")

        node_data = self.code_graph.get_node(node_id)
        if not node_data or 'code' not in node_data:
            print(f"❌ Node {node_id} not found or has no executable code.")
            return

        # 💡 def: create and load entire combined str code base executable in ray.remote -> run function
        # NOTE: We only send the function's own code. To run the *entire* codebase,
        # the codebase would have to be combined/imported within the remote function.
        function_code = node_data['code']

        # Execute remotely (wrapped func)
        remote_task = self._execute_code_remote.remote(self, function_code, input_params)

        # Get result (blocks until finished)
        result = ray.get(remote_task)
        print(f"Final Remote Result: {result}")
        return result


# --- Execution Block ---

def testing_def(cli: CodeAnalyzerCLI):
    """// Defines and executes the step-by-step workflow for testing."""
    # Create required placeholder files for robust testing
    with open(REQUIREMENTS_FILE, 'w') as f:
        f.write("numpy==1.24\nray==2.10\njax==0.4")
    with open(ENV_FILE_PATH, 'w') as f:
        f.write("T=1.0\nMASS=0.1\nG_COUPLING=0.6")
    with open("example_module.py", 'w') as f:
        f.write("def method_a(x: float, y: jnp.ndarray): return x * jnp.sum(y)")

    print("\n\n###################################################################")
    print("## 🧪 STARTING END-TO-END WORKFLOW EXECUTION 🧪 ##")
    print("###################################################################")

    # STEP 1: Graph Construction
    print("--- STEP 1: GRAPH CONSTRUCTION ---")
    cli.create_nx_graph()

    # STEP 2: CLI Action: Serialization
    print("\n--- STEP 2: SERIALIZATION ---")
    json_path = cli.cli_action_create_json()

    # STEP 3: CLI Action: Semantic Search and Pathfinding
    print("\n--- STEP 3: SEMANTIC SEARCH & PATHFINDING ---")
    # Search for a function that performs "multiplication" (using the embedding score)
    query = "function that performs array multiplication"
    related_nodes = cli.cli_action_sem_search_and_path(query)
    print(f"Top related nodes and dependencies: {related_nodes}")

    # STEP 4: CLI Action: Remote Execution
    print("\n--- STEP 4: REMOTE EXECUTION ---")
    # Running a method that was identified
    if "method_a" in cli.code_graph.G.nodes:
        # Example: Running the remote method with parameters from .env and input
        cli.cli_action_run_remote(node_id="method_a", input_params={"x": 5.0, "y": jnp.array([1, 2, 3])})
    else:
        print("Skipping remote run: Target method 'method_a' not found in the test graph.")

    # STEP 5: Final Cleanup (Optional: Pyvis)
    print("\n--- STEP 5: VISUALIZATION (Pyvis Simulation) ---")
    # Pyvis requires an HTML output, which is complex in this environment.
    # We simulate the generation by confirming graph existence.
    print(f"Pyvis Visualization could be generated using the graph from: {json_path}")


if __name__ == '__main__':
    cli = CodeAnalyzerCLI(codebase_root=CODEBASE_ROOT)
    testing_def(cli)




USE CODE SCHEMA:
import json
import os
import time
from tempfile import TemporaryDirectory
from typing import List, Dict
import networkx as nx
import queue

from qf_utils.all_subs import ALL_SUBS
from utils._np.serialize_complex import check_serialize_dict
from graph_visualizer.pyvis_visual import create_g_visual
from utils.manipulator import Manipulator
from utils.queue_handler import QueueHandler
from utils.utils import Utils

class GUtils(Utils):
    """
    Handles State G local and
    History G through DataManager

    ALERT:
    DB Pushs need to be ahndled externally (DBManager -> _google)
    """

    def __init__(
            self,
            user_id,
            G=None,
            g_from_path=None,
            nx_only=False,
            # queue: queue.Queue or None = None,
            enable_data_store=True,
            history_types=None,
            file_store=None
    ):
        super().__init__()
        self.G = None
        self.enable_data_store = enable_data_store
        self.g_from_path = g_from_path
        self.get_nx_graph(G)
        self.nx_only = nx_only
        self.history = {}
        self.user_id = user_id

        #todo just temporary look for demo G in QFS and BB
        demo_G_save_path = r"C:\Users\wired\OneDrive\Desktop\Projects\qfs\demo_G.json" if os.name == "nt" else "demo_G.json"
        if os.path.isfile(demo_G_save_path):
            self.demo_G_save_path = demo_G_save_path
        else:
            self.demo_G_save_path = r"C:\Users\wired\OneDrive\Desktop\qfs\demo_G.json" if os.name == "nt" else "demo_G.json"

        self.manipulator = Manipulator()
        self.q_handler = QueueHandler(queue)

        if self.enable_data_store is True:
            self.datastore = nx.Graph()
            self.history_types = history_types  # list of nodetypes captured by dataqstore  ALL_SUBS + ["ENV"]

        self.file_store=file_store or TemporaryDirectory()

        self.metadata_fields = [
            "graph_item",
            "index",
            "entry_index",
            "time",
        ]

        # Sim timestep must be updated externally for each loop
        self.timestep = None
        self.key_map = set()
        self.id_map = set()
        self.schemas = {}
        print("GUtils initialized")

    ####################################
    # CORE                             #
    ####################################

    def get_node_attrs_core(
            self,
            nid,
            ntype,
            parent
    ):
        return dict(
            nid=nid,
            tid=0,
            type=ntype,
            #parent=[parent]
        )

    def get_nodes_each_type(self, ref_ntypes:list[str]):
        all_ntypes = {}
        for nid, attrs in self.G.nodes(data=True):
            ntype = attrs.get("type")
            if ntype not in all_ntypes and ntype in ref_ntypes:
                all_ntypes[ntype] = attrs
        print("All nodes etracted")
        return all_ntypes



    def get_edge(self, src, trgt):
        return self.G.edges[src, trgt]

    def get_graph(self):
        return self.G
    #graph engine

    def get_node(self, nid):
        try:
            return self.G.nodes[nid]
        except:
            return None

    def print_edges(self, trgt_l, src_l):
        print("len edges", len([
            attrs
            for src, trgt, attrs in self.G.edges(data=True)
            if attrs.get("src_layer").upper() == src_l.upper()
            and attrs.get("trgt_layer").upper() == trgt_l.upper()
        ]))

    def add_node(self, attrs: dict, timestep=None, flatten=False):
        #print("add_node:", attrs)
        attrs = self.manipulator.clean_attr_keys(
            attrs,
            flatten
        )

        if attrs.get("type") is None:
            print("NEW NODE ATTRS")
            # #pprint.pp(attrs)

        attrs["type"] = attrs["type"].upper()
        nid = attrs["nid"]

        if self.nx_only is False:
            self.local_batch_loader(attrs)

        self.G.add_node(nid, **{k: v for k, v in attrs.items() if k != "id"})

        # Add history entry
        self.h_entry(nid, {k: v for k, v in attrs.items() if k != "id"})

        # Extedn keys
        self._extend_key_map(attrs)
        self._extend_id_map(nid)
        return True

    def h_entry(self, nid, attrs, timestep=None, graph_item="node"):
        ntype = attrs.get("type", "")
        if ntype is None:
            ntype = graph_item  # -> SET EDGE

        if self.enable_data_store is True:
            if timestep is None:
                timestep = attrs.get("time", 0)

            history_id = f"{nid}_{int(time.time())}_{timestep}"

            len_type_entries = len(
                [
                    (inid, iattrs)
                    for inid, iattrs in self.datastore.nodes(data=True) if
                    iattrs.get("type", "0").upper() == attrs.get("type", "1").upper()
                ]
            )

            attrs = dict(
                type=nid,
                entry_index=len_type_entries,
                graph_item=graph_item,
                base_type=ntype,
                **{k: v for k, v in attrs.items() if k not in ["id", "type"]}
            )

            #print("Add H Entry:")
            ##pprint.pp(attrs)

            # Extedn keys
            self._extend_key_map(attrs)
            self._extend_id_map(nid)

            self.datastore.add_node(
                history_id,
                **attrs
            )
            #print("H entry node added", self.datastore.nodes[history_id])
        else:
            raise ValueError("Invalid data!!!!", nid, attrs)

    def add_edge(
            self,
            src=None,
            trgt=None,
            attrs: dict or None = None,
            flatten=False, timestep=None,
            index=None
    ):
        #print("add_edge:", locals())
        color = None
        # Check
        if index is None:
            index = attrs.get("index", None)
        if index is not None:
            color = f"rgb({index + .5}, {index + .5}, {index + .5})"
        # #print("color set:", color)

        try:
            src_layer = self.manipulator.replace_special_chars(attrs.get("src_layer")).upper()
            trgt_layer = self.manipulator.replace_special_chars(attrs.get("trgt_layer")).upper()

            # #print("src_layer", src_layer)
            # #print("trgt_layer", trgt_layer)
            if src is None:
                src = attrs.get("src")
            if trgt is None:
                trgt = attrs.get("trgt")

            if src and trgt and src_layer and trgt_layer:
                if isinstance(src, int):
                    src = str(src)
                if isinstance(trgt, int):
                    trgt = str(trgt)
                # #print("int conv...")

                attrs = self.manipulator.clean_attr_keys(attrs, flatten)
                # #print("attrs_new", attrs )
                rel = attrs["rel"].lower().replace(" ", "_")

                edge_id = f"{src}_{rel}_{trgt}"

                attrs = {
                    **attrs,
                    "src": src,
                    "trgt": trgt,
                    "eid": edge_id,
                    "tid": 0,
                    "color": color,
                }

                # Add keys
                self._extend_key_map(attrs)
                self._extend_id_map(attrs["eid"])

                # #print(f"ids {src} -> {trgt}; Layer {src_layer} -> {trgt_layer}")
                edge_table_name = f"{src_layer}_{rel}_{trgt_layer}"
                attrs["type"] = edge_table_name

                src_node_attr = {"nid": src, "tid": 0, "type": src_layer}
                trgt_node_attr = {"nid": trgt,"tid": 0, "type": trgt_layer}
                # #print(f"Add {src} -> trgt: {trgt}")

                if self.nx_only is False:
                    # todo run in executor
                    # #print("Upsert Local Batch Loader")
                    self.local_batch_loader(src_node_attr)
                    self.local_batch_loader(trgt_node_attr)
                    self.local_batch_loader(attrs)

                # #print("Upsert to NX")
                self.G.add_edge(src, trgt, **{k: v for k, v in attrs.items()})
                self.G.add_node(src, **src_node_attr)
                self.G.add_node(trgt, **trgt_node_attr)

                # Add history entry
                self.h_entry(
                    nid=attrs["eid"],
                    attrs={k: v for k, v in attrs.items() if k != "id"},
                    graph_item="edge"
                )
                return attrs
            else:
                raise ValueError(f"Wrong edge fromat")

        except Exception as e:
            raise ValueError(f"Skipping link src: {src} -> trgt: {trgt} cause:", e, attrs)

    def _extend_key_map(self, attrs):
        for k in list(attrs.keys()):
            if k not in self.key_map:
                self.key_map.add(k)


    def _extend_id_map(self, nid):
        if nid not in self.id_map:
            self.id_map.add(nid)

    def get_edges(self, datastore=True, just_id=False):
        if datastore is False:
            if just_id is True:
                edges = [attrs.get("id") for _, _, attrs in self.G.edges(data=True)]
            else:
                edges = [{
                    "src": src,
                    "trgt": trgt,
                    "attrs": attrs
                }
                    for src, trgt, attrs in self.G.edges(data=True)
                ]

        else:
            edges = [{"attrs": attrs} for eid, attrs in self.datastore.edges(data=True) if
                    attrs.get("graph_item").lower() == "edge"]
        return edges

    def get_edges_from_node(self, nid, datastroe=True):
        new_all_edges = []

        if datastroe is False:
            all_edges = [{"src": src, "trgt": trgt, "attrs": attrs} for src, trgt, attrs in self.G.edges(data=True)]
            for edge in all_edges:
                if edge["src"] == nid or edge["trgt"] == nid:
                    new_all_edges.append(edge)
        else:
            return [{"attrs": attrs, "eid": eid} for eid, attrs in self.datastore.edges(data=True) if
                    attrs.get("graph_item").lower() == "edge"]

        if len(new_all_edges):
            all_edges = new_all_edges

        return  all_edges


    def update_node(self, attrs, disable_history=False):
        nid = attrs.get("nid")
        node_attrs = self.G.nodes[nid]
        if node_attrs is None:
            print("Node couldnt be updated...")
            return

        # todo serilize @ save
        #attrs = check_serialize_dict(attrs, [k for k in attrs.keys()])

        # Add keys
        self._extend_key_map(attrs)

        self.G.nodes[nid].update(attrs)

        if self.enable_data_store is True and disable_history is False:
            # Add history entry
            self.h_entry(
                attrs["nid"],
                {k: v for k, v in attrs.items() if k != "nid"},
                graph_item="node"
            )

    def update_edge(self, eid, attrs, src=None, trgt=None, rels: str or list = None, temporal=False):
        # rel = attrs.get("rel", "").lower().replace(" ", "_")
        """
        src_layer = attrs.get("src_layer").upper()
        trgt_layer = attrs.get("trgt_layer").upper()
        table_name = f"{src_layer}_{rel}_{trgt_layer}
        """



        # serialize attrs
        # todo @ save chek serilize otherwise ray actors get serialized fuck in
        # attrs = check_serialize_dict(attrs, [k for k in attrs.keys()])

        # Add keys
        self._extend_key_map(attrs)

        # Update nx
        if "MultiGraph" in str(type(self.G)):
            for key, edge in self.G.get_edge_data(src, trgt).items():
                erel = edge.get("rel")
                if erel in rels:
                    if self.enable_data_store is True:
                        edge_id = f"{src}_{erel}_{trgt}"
                        self.h_entry(
                            edge_id,
                            {k: v for k, v in attrs.items() if k != "eid"},
                            graph_item="edge"
                        )
                    self.G.edges[src, trgt, key].update(attrs)
        else:
            if self.enable_data_store is True:
                self.h_entry(
                    "edge_id",
                    {k: v for k, v in attrs.items() if k != "eid"},
                    graph_item="edge"
                )
            self.G.edges[src, trgt].update(attrs)

        # todo handle async rt spanner || fbrtdb

    ####################################
    # HELPER
    ####################################

    def get_nx_graph(self, G):
        if self.g_from_path is not None:
            if os.path.exists(self.g_from_path):
                self.load_graph()
        if G is not None:
            self.G = G
        elif self.G is None:
            self.G = nx.MultiGraph()  # normaler G da gluon -> gluon sonst explodieren würde
        #print("Local Graph loaded")

    def save_graph(self, dest_file, ds=False):
        print("Save Gs")
        if ds is True:
            G=self.datastore
        else:
            G=self.G
        self._link_safe(
            G,
            dest_file
        )
        print(f"G data written to :{dest_file}")


    def _link_safe(self, G, dest_name):
        self.check_serilize(G)
        data = nx.node_link_data(G)

        with open(f"{dest_name}", "w") as f:
            json.dump(data, f)

    def check_serilize(self, G):
        # srialize
        for nid, attrs in G.nodes(data=True):
            G.nodes[nid].update(
                check_serialize_dict(
                    attrs,
                    [k for k in attrs.keys()],
                )
            )
        for src, trgt, attrs in G.edges(data=True):
            G.edges[src, trgt].update(
                check_serialize_dict(
                    attrs,
                    [k for k in attrs.keys()],
                )
            )
        return G











    def load_graph(self, local_g_path=None):
        if local_g_path is None:
            local_g_path = self.g_from_path
        """Loads the networkx graph from a JSON file."""
        print(f"📂 Loading graph from {local_g_path}...")
        with open(local_g_path, "r", encoding="utf-8") as f:
            graph_data = json.load(f)  # Use json.load() for files, not json.loads()

        self.G = nx.node_link_graph(graph_data)

        # return env
        for k, v in self.G.nodes(data=True):
            type = v.get("type")
            if type == "ENV":
                return k, v
        print(f"✅ Graph loaded! {len(self.G.nodes)} nodes, {len(self.G.edges)} edges.")

    def print_status_G(self):
        print("STATUS:", self.G)
        everything = {}
        for k, v in self.G.nodes(data=True):
            ntype = v.get("type")
            if ntype not in everything:
                everything[ntype] = 0
            everything[ntype] += 1

        for k, v in everything.items():
            print(f"{k}:{v}")

    def local_batch_loader(self, args):
        table_name = args.get("type")
        row_id = args.get("nid", args.get("eid"))
        if table_name:
            if table_name not in self.schemas:
                self.schemas[table_name] = {
                    "schema": {},
                    "rows": [],
                    "id_map": set(),
                }
                #print(f"Added {table_name} to schema")

            if row_id not in [item for item in self.schemas[table_name]["id_map"]]:
                # #print(f"Insert {row_id} into {table_name}")
                self.schemas[table_name]["rows"].append(args)
                self.schemas[table_name]["id_map"].add(row_id)
            # else:
            # #print(f"{row_id} already in schema")
        # #print("Added args")

    def get_single_neighbor_nx(self, node, target_type:str):
        try:
            if isinstance(node, tuple):
                node = node[0]
            for neighbor in self.G.neighbors(node):
                if self.G.nodes[neighbor].get('type') == target_type:
                    return neighbor, self.G.nodes[neighbor]
            return None, None  # No neighbor of that type found
        except Exception as e:
            print(f"Couldnt fetch content: {e}")

    def get_node_list(self, trgt_types, just_id=False):
        interest = {
            nid: attrs
            for nid, attrs in self.G.nodes(data=True)
            if attrs.get("type") in trgt_types
        }
        if just_id is True:
            interest = list(interest.keys())
        return interest


    def get_edge_ids(self, src, neighbor_ids):
        eids = []
        for nnid in neighbor_ids:
            eattrs = self.G.get_edge_data(src, nnid)
            if "eid" in eattrs:
                eid = eattrs["nid"]
            else:
                rel = eattrs.get("rel")
                eid = f"{src}_{rel}_{nnid}"
            eids.append(eid)
        #print(f"Edge Ids extracted: {eids}")
        return eids





    def get_neighbor_list(
            self,
            node,
            target_type: str or list or None = None,
            just_ids=False
    ) -> List[str] or Dict[str, Dict]:
        neighbors = {}

        # Filter Input
        if isinstance(target_type, str):
            target_type = [target_type]
        upper_trgt_types = [t.upper() for t in target_type]

        if just_ids is True:
            nids = list(self.G.neighbors(node))
            #print(f"Node Ids extracted: {nids}")
            return nids

        for neighbor in self.G.neighbors(node):
            # Get neighbor from type
            nattrs = self.G.nodes[neighbor]
            if target_type is not None:
                ntype = nattrs.get('type')
                if ntype in upper_trgt_types:
                    if neighbor not in neighbors:
                        neighbors[neighbor] = {}
                    neighbors[neighbor] = nattrs

        #print(f"Neighbors extracted: {neighbors.keys()}")
        return neighbors




    def get_neighbor_list_rel(
            self,
            node:str,
            trgt_rel: str or list or None = None,
            as_dict=False
    ):
        neighbors = {}
        edges = {}

        if isinstance(trgt_rel, str):
            trgt_rel = [trgt_rel]

        # Get neighbor from rel
        for nnid in self.G.neighbors(node):
            edge_data = self.G.get_edge_data(node, nnid)

            if isinstance(self.G, (nx.MultiGraph, nx.MultiDiGraph)):
                for key, edge_attrs in edge_data:
                    ntype = edge_attrs.get('type')

                    if edge_attrs.get("rel") in trgt_rel:
                        if ntype not in neighbors:
                            neighbors[nnid] = {}
                        edges[nnid] = edge_attrs
            else:
                # check if rel matches
                if edge_data.get("rel").lower() in [rel.lower() for rel in trgt_rel]:
                    # get nodes from extracted edges
                    attrs = self.G.nodes[nnid]
                    neighbors[nnid] = {
                        "nid": nnid,
                        **{
                            k: v
                            for k, v in attrs.copy().items()
                            if k != "nid"
                        }
                    }

        print("get_neighbor_list_rel")
        #pprint.pp(neighbors)

        if as_dict is True:
            return neighbors
        return [
            (nid, attrs)
            for nid, attrs in neighbors.items()
        ]


    def remove_node(self, node_id, ntype):
        for row in self.schemas[ntype]["rows"]:
            if row["nid"] == node_id:
                self.schemas[ntype]["rows"].remove(row)
                break
        self.G.remove_node(node_id)


    def cleanup_self_schema(self):
        # #print("Cleanup schema")
        for k, v in self.schemas.items():
            v["rows"] = []


    def build_G_from_data(
            self,
            initial_data,
            env_id=None,
            save_demo=False,
    ):
        # --- Graph aufbauen ---
        env = None
        data_keys = [k for k in initial_data.keys()]
        print(f"INITIAL DATA KEYS: {data_keys}")

        for node_type, node_id_data in initial_data.items():
            # Just get valid
            nupper = node_type.upper()
            valid_types = [*ALL_SUBS, "PIXEL", "ENV", "EDGES"]
            nupper_valid_t = nupper in valid_types

            print(f"{nupper} valid: {nupper_valid_t}")

            if nupper_valid_t:
                if isinstance(node_id_data, dict):  # Sicherstellen, dass es ein Dictionary ist
                    for nid, attrs in node_id_data.items():
                        # print(f">>>NID, {nid}")
                        if node_type.lower() == "EDGES":
                            parts = nid.split(f"_{attrs.get('rel')}_")
                            # print("parts", parts)
                            # check 2 ids in id and
                            if len(parts) >= 2:
                                self.add_edge(
                                    parts[0],
                                    parts[1],
                                    attrs=attrs
                                )
                            else:
                                print("something else!!!")

                        elif node_type == "ENV":
                            print("Env recognized")
                            env = attrs
                            env_id = nid
                            self.add_node(
                                attrs=attrs,
                            )
                            # Speichern Sie die env_id, falls benötigt
                        else:
                            self.add_node(
                                attrs=attrs,
                            )
                else:
                    print(f"DATA NOT A DICT:{node_type}:{node_id_data}")
                    # #pprint.pp(node_id_data)
                # time.sleep(10)

            else:
                print(f"TYPE NOT VALID:{node_type}")

        print(f"Graph successfully build: {self.G}")

        if save_demo is True and getattr(self, "demo_G_save_path", None) is not None:
            self.save_graph(dest_file=self.demo_G_save_path)
        return env, env_id

    def delete_node(self, delid):
        if delid and self.G.has_node(delid):
            #print(f"Del node {delid}")
            self.G.remove_node(delid)
        else:
            print(f"Couldnt delete since {delid} doesnt exists")


    def get_node_pos(self, G=None):
        if G==None:
            G = self.G
        serializable_node_copy = []
        valid_types = [*ALL_SUBS, "PIXEL"]
        for nid, attrs in G.nodes(data=True):
            ntype = attrs.get("type")
            if ntype in valid_types:
                # todo single subs
                serializable_node_copy.append(
                    {
                        "nid": nid,
                        "pos": attrs.get("pos")
                    }
                )
        return serializable_node_copy


    def get_nodes(
            self,
            filter_key=None,
            filter_value:str or list=None,
            just_id=False,
    ) -> list[tuple] or list[str]:
        nodes = self.G.nodes(
            data=True
        )
        print(f"len nodes: {len(nodes)}")
        if filter_key is not None and filter_value is not None:
            new_nodes = []
            if not isinstance(filter_value, list):
                filter_value = [filter_value]

            for nid, attrs in nodes:
                if attrs.get(filter_key) in filter_value:
                    if just_id is True:
                        new_nodes.append(
                            nid
                        )
                    else:
                        new_nodes.append(
                            (nid, attrs)
                        )
            nodes = new_nodes

        return nodes


    def get_edges_src_trgt_pos(self, G=None, get_pos=False) -> list[dict]:
        if G == None:
            G = self.G
        edges=[]
        valid_types = [*ALL_SUBS, "PIXEL"]
        for src, trgt, attrs in G.edges(data=True):
            src_attrs = G.nodes[src]
            trgt_attrs = G.nodes[trgt]

            src_type = src_attrs["type"]
            trgt_type = trgt_attrs["type"]

            if src_type in valid_types and trgt_type in valid_types:
                if get_pos is True:
                    src_pos = src_attrs["pos"]
                    trgt_pos = trgt_attrs["pos"]

                    # todo calc weight based on
                    edges.append(
                        dict(
                            src=src_pos,
                            trgt=trgt_pos
                        )
                    )
                else:
                    edges.append(
                        dict(
                            src=src,
                            trgt=trgt
                        )
                    )
        #print(f"edge src trgt pos set: {edges}")
        return edges

    def create_html(self):
        save_path = os.path.join(
            self.file_store.name,
            "graph.html",
        )
        html = create_g_visual(self.datastore, dest_path=None)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"HTML Graph was written to: {save_path}")





    def categorize_nodes_in_types(self, valid_ntypes=ALL_SUBS) -> dict[list]:
        categorized = {}
        for nid, attrs in self.G.nodes(data=True):
            ntype = attrs.get("type")
            if ntype:
                ntype=ntype.upper()
            if ntype in [n.upper() for n in valid_ntypes]:
                if ntype not in categorized:
                    categorized[ntype] = []
                categorized[ntype].append(
                    (nid, attrs)
                )
        print("Nodes in types categorized")
        return categorized

    def categorize_nodes_in_qfns(self) -> dict[list[tuple]]:
        categorized = {}
        points = [(nid, attrs) for nid, attrs in self.G.nodes(data=True) if attrs.get("type") == "PIXEL"]

        for qfn in points:
            qfn_id = qfn[0]
            categorized[qfn_id] = self.get_neighbor_list_rel(qfn_id, trgt_rel="has_field")

        print("Nodes in PIXELs categorized")
        return categorized


    ###################
    # GET
    ###################

    def get_demo_G_save_path(self):
        return self.demo_G_save_path

    def get_env(self):
        """env:tuple = [(nid, attrs) for nid, attrs in self.G.nodes(data=True) if attrs.get("type") == "ENV"][0]
        return {"id": env[0], **{k:v for k,v in env[1].items() if k != "id"}}"""
        for nid, attrs in self.G.nodes(data=True):
            if attrs.get("type") == "ENV":
                print("ENV entry found")
                return {
                    "nid": nid,
                    **{k: v for k, v in attrs.items() if k != "nid"}}

