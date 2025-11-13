# firecoder

synthax: 
"::" = next jump in local wf
"->" = next pathway step
"-->" = edge connect 
">" bring workflows (wf) = together
"- receive code str ->" = rcs
"related" - <>
"receive list prompt" = rlp
"rlp -> loop graph to find components of interest (higher ss score then .9)" -> COLLECTOR

legend
similairity search = ss
vectro store = vs

ROLE:
SENIOR SCIENTIFIC SOFTWARE ARCHITECT with the task to build software systems


PROMPT:
Improve the prmpt qualiy followign the 
GOAL:
Provide an entire codebase genrated with your imroved prompt withing a single file

**CRITERIA:**
Clarity and Specificity 💡
Use Direct Imperatives
Specify the Format
Define Constraints (The "Don't" List)
Set the Length and Tone
Provide Examples (Few-Shot Prompting)
Clarify Necessary Knowledge (Context)
Use Chain-of-Thought (CoT)



CONTEXT:

# Relay remote
- receive query string -> create 5 versions of prompt using a gem(genai api call - load env keys)  > classifier rlp to dict[functionality(, runnable def]
- qa handling for actions that has been performed. add here qa functionaity in harmony to the terminal input field (render)
   
# Globac
**info** data store-node for entire configuration objects, logs, outputs -> apply outputs to specific nodes ids


# Researcher 
- - define local vs
  - define static search engine prompt: gather information based specific isntructions to support the underlying goal
  - fetch 
- rlp -> convert to 5 search queries -> gsearch request -> embed and save page content first 5 pages (non ads) in table format within local vs. -> extract: pip packages required for this task -> run subprocess -> load packages in Graph (nid=package_name, ref=package_instance)


# ADD_CODE_GRAPH 
 use ast: identify all present datatypes in each file -> extract and classify file content -> add_node for each with the parent=[parent file-node] and type=datatype(class, def, comment) ´, embed datatype (class/method etc) (only node keys wrapped inside dict(embedding(364dim), nid, t, type), code make available
- collect all packages specified in r.txt -> add_node(dict(nid, package instance, embed description) 
- scan each datatype for sub modules/classes/functions/motheds/packages used from outer space -> link logical to destination(e.g. method A defines a class inside from external) -> add_edge with rel="uses" - classify directions (seppare between imports or imported) -> nx.link_node_data & pyvis render -> save files (json & html) in root -> print_status_G


# Graph Engine remote
**info** act as globalvailable graph remote
- - create nx.MiltiDiGraph 
- walk local dir (exclude .venv) -> add_node all folders type = MODULE, description=embed(llm file request: sum content within  -> extract content each file add_node file_name -> add_edge file_name --> module rel=has_file, ADD_CODE_GRAPH

# Edit
COLLECTOR -> llm request with specific code - attr values as input to adapt (anwer must be altimes ready to use adapted code) -> write changes to Graph  

# Create
- rlp -> gem api call: create code base -> ADD_CODE_GRAPH

# Run
COLLECTOR -> loop graph code content for programming language -> create suitable env (e.g. for js: compile code (webpack) -> start local server -> open browser to render changes OR py: figure out the main method -> run it) -> write output to Globac

# qa 
- - static_prmpt: answer the questions based on user query
COLLECTOR -> llm call(static prompt)




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
import subprocess
import time
import sys
import queue
from itertools import chain

# --- Configuration & Setup ---

# one-liner comment: Initialize Ray if not already running (required for remote execution).
if not ray.is_initialized():
    # include the entire setup to init and run ray
    ray.init(ignore_reinit_error=True)
    print("💡 Ray initialized.")

# --- Constants & Global Types ---
CODEBASE_ROOT = r"./"  # Use local root for generic testing
EXCLUDE_DIRS = {".venv", "__pycache__", ".git"}
ENV_FILE_PATH = "project.env"
REQUIREMENTS_FILE = "r.txt"
OUTPUT_GRAPH_JSON = "code_graph.json"

# NOTE: LLM/Embedding simulation parameters
LLM_EMBEDDING_DIM = 364  # Per improved prompt context
SIMILARITY_THRESHOLD = 0.9


# --- Utility Functions ---

# one-liner comment: Extracts docstring from function or class node.
def _get_docstring(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> str:
    """// Extracts docstring."""
    return ast.get_docstring(node) or ""


# one-liner comment: Extracts type name from annotation or defaults to 'Any'.
def _get_type_name(node: Optional[ast.expr]) -> str:
    """// Extracts type name."""
    return ast.unparse(node) if node else 'Any'


# one-liner comment: Loads environment variables from the specified path.
def load_env_variables(path: str) -> Dict[str, str]:
    """// Loads environment variables."""
    # MOCK: Simplified loading for testing
    if os.path.exists(path):
        # for all params used(method/class header) by specific datatype: collect data from .env file (open...)
        return {"T": "1.0", "MASS": "0.1", "G_COUPLING": "0.6"}
    return {}


# one-liner comment: Loads package names from r.txt.
def load_requirements(path: str) -> List[str]:
    """// Loads package names."""
    if os.path.exists(path):
        with open(path, 'r') as f:
            return [line.strip().split('==')[0] for line in f if line.strip() and not line.startswith('#')]
    return []


# --- Embedding and LLM Simulation ---

# one-liner comment: Simulates the creation of a 364-dimensional embedding vector.
def embed_text_sim(text: str) -> np.ndarray:
    """// embed datatype (class/method etc) (only node keys wrapped inside dict(embedding(364dim)..."""
    random_seed = sum(ord(c) for c in text) % 1000
    np.random.seed(random_seed)
    return np.random.rand(LLM_EMBEDDING_DIM)


# one-liner comment: Calculates Cosine Similarity between two embeddings (Simulated).
def get_similarity_score_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """// Berechnet die Ähnlichkeit."""
    dot_product = np.dot(emb1, emb2)
    norm_a = np.linalg.norm(emb1)
    norm_b = np.linalg.norm(emb2)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.clip(dot_product / (norm_a * norm_b), 0.0, 1.0)


# one-liner comment: Handles JSON serialization for complex objects.
def custom_serializer(obj):
    """
    // MANDATORY: Handle complex data types by converting them to JSON-serializable strings.
    // Converts NumPy/JAX types to strings/lists for serialization.
    """
    if isinstance(obj, np.ndarray):
        # Convert array to serializable structure
        return {'__ndarray__': True, 'dtype': obj.dtype.name, 'shape': obj.shape, 'data': obj.tolist()}
    elif isinstance(obj, np.dtype):
        return obj.name  # Convert dtype object to string
    elif isinstance(obj, tuple):
        return list(obj)  # Convert shape tuple to list
    elif isinstance(obj, (ActorHandle, ObjectRef)):
        return str(obj)  # Convert Ray references to string
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# one-liner comment: Placeholder for LLM calls (e.g., Gemini/Local LLM).
def llm_call_sim(prompt: str, task: str) -> str:
    """// Simulates a remote LLM call for complex tasks."""
    if task == "classify": return "MODULE"
    if task == "create_code": return "# Generated Code by Creator\ndef generated_function(): return 100"
    if task == "edit_code": return "print('Code Edited Successfully!')"
    if task == "debug_fix": return "fixed_code = 'print(\"Error fixed!\")'"
    return f"LLM_RESPONSE_{task}"


# --- Graph Utilities (Adapted from GUtils Schema) ---

class Manipulator:
    """// Helper class to encapsulate cleaning functions."""

    # one-liner comment: Sanitizes keys by keeping only alphanumeric and underscore characters.
    def replace_special_chars(self, s: Union[str, bytes]) -> str:
        """// Sanitizes keys."""
        if not isinstance(s, (str, bytes)): return ""
        return re.sub(r'[^a-zA-Z0-9_]', '', s)

    # one-liner comment: Cleans dictionary keys for serialization.
    def clean_attr_keys(self, attrs: Dict[Any, Any], flatten=False) -> Dict[str, Any]:
        """// Cleans the keys of a dictionary."""
        cleaned_attrs = {}
        for k, v in attrs.items():
            cleaned_key = self.replace_special_chars(k)
            cleaned_attrs[cleaned_key] = v
        return cleaned_attrs


class LocalGraphUtils:
    """// Graph utility class using NetworkX MultiDiGraph (Transactional Core)."""

    # one-liner comment: Initializes the MultiDiGraph and helper attributes.
    def __init__(self):
        # 🚀 create nx.MiltiDiGraph
        self.G = nx.MultiDiGraph()
        self.manipulator = Manipulator()
        self.pathway_nodes: Set[str] = set()
        print("🧭 Graph Initialized (MultiDiGraph)")

    # one-liner comment: Adds a node, computes embedding, and ensures clean attributes (ADD_CODE_GRAPH atomic action).
    def add_node(self, attrs: Dict[str, Any], timestep=None, flatten=False):
        """// Adds a node with sanitized attributes."""
        # check avoid errors
        if attrs.get("type") is None: attrs["type"] = "GENERIC_NODE"
        attrs["type"] = attrs["type"].upper()
        nid = attrs["nid"]

        # 💡 Generiere Embedding
        code_content = attrs.get('code', attrs.get('nid', ''))
        attrs['embedding'] = embed_text_sim(code_content).tolist()

        # Clean, add timestamp (time field), and execute transaction
        attrs["time"] = time.time()
        cleaned_attrs = self.manipulator.clean_attr_keys(attrs, flatten)

        # NOTE: In a real system, a LOCK would be acquired here.
        self.G.add_node(nid, **{k: v for k, v in cleaned_attrs.items() if k != "id"})
        # LOCK released here.
        return True

    # one-liner comment: Adds a directed edge with sanitized attributes (ADD_CODE_GRAPH atomic action).
    def add_edge(
            self,
            src=None,
            trgt=None,
            attrs: dict or None = None,
            flatten=False, timestep=None,
            index=None
    ):
        """// Adds a directed edge."""
        attrs = attrs if attrs is not None else {}
        if src and trgt and attrs.get("rel"):
            # Ensure nodes exist
            if not self.G.has_node(src): self.G.add_node(src, nid=src, type="UNKNOWN_SRC")
            if not self.G.has_node(trgt): self.G.add_node(trgt, nid=trgt, type="UNKNOWN_TRGT")

            attrs["time"] = time.time()
            cleaned_attrs = self.manipulator.clean_attr_keys(attrs, flatten)

            # NOTE: LOCK acquired here.
            self.G.add_edge(src, trgt, **cleaned_attrs)
            # LOCK released here.
            return True
        return False

    # one-liner comment: Retrieves a node's attributes.
    def get_node(self, nid):
        """// Retrieves a node's attributes."""
        try:
            return self.G.nodes[nid]
        except:
            return None

    # one-liner comment: Pathfinding algorithm (BFS) for dependency retrieval (Collector remote logic).
    def get_neighbors_recursive(self, start_node: str, relationship: str = "uses") -> Set[str]:
        """// Pathfinding algorithm (BFS) to find dependencies (AAA)."""
        # loop nids with sc > .9: include a pathfinding algirithmus whcih receivesa nid of a specific datatype
        self.pathway_nodes.clear()
        queue_list = [start_node]

        while queue_list:
            current_node = queue_list.pop(0)
            if current_node in self.pathway_nodes:
                continue

            self.pathway_nodes.add(current_node)

            # 💡 Relational check: needs, uses, has_method, has_class
            for _, neighbor, edge_attrs in self.G.out_edges(current_node, data=True):
                relationship_type = edge_attrs.get('rel')

                # next jump in local wf :: enqueue neighbor
                if relationship_type in [relationship, 'needs', 'uses', 'has_method', 'has_class']:
                    queue_list.append(neighbor)

        return self.pathway_nodes


class StructInspector(ast.NodeVisitor):
    """// Traverses AST to map code structure and dependencies (ADD_CODE_GRAPH logic)."""

    # one-liner comment: Initializes the inspector with graph access and file context.
    def __init__(self, lex: LocalGraphUtils, module_name: str, file_content: str):
        self.current_class: Optional[str] = None
        self.lex = lex
        self.module_name = module_name
        self.file_content = file_content
        # Add root file node
        self.lex.add_node({"nid": module_name, "type": "FILE_MODULE", "code": file_content})

    # one-liner comment: Visits Class definitions, adds nodes, and links to the module.
    def visit_ClassDef(self, node: ast.ClassDef):
        # 1. Add CLASS Node
        self.current_class = node.name
        self.lex.add_node({"nid": node.name, "parent": [self.module_name], "type": "CLASS", "code": ast.unparse(node)})
        # 2. Link MODULE -> CLASS --> edge connect
        self.lex.add_edge(src=self.module_name, trgt=node.name, attrs={'rel': 'has_class'})
        self.generic_visit(node)
        self.current_class = None

    # one-liner comment: Processes function/method definitions, adding nodes, parameters, and needs links.
    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        """// Processes methods/functions, parameters, and return types."""
        method_id = node.name
        entire_def = ast.unparse(node)
        parent_id = self.current_class or self.module_name

        # 1. Add METHOD Node
        self.lex.add_node({
            "nid": method_id,
            "parent": [parent_id],
            "type": "METHOD",
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
                attrs=dict(rel='needs', type=param_type))

        self.generic_visit(node)

    # one-liner comment: Scans for function/method calls (usages) and links them as 'uses'.
    def visit_Call(self, node: ast.Call):
        """// Scans for functions/methods used from outer space."""
        # Simplified call resolution
        called_name = ast.unparse(node.func).split('.')[-1]

        if called_name:
            current_component = self.current_class or self.module_name
            # Link current component -> used function/method
            self.lex.add_edge(
                src=current_component,
                trgt=called_name,
                attrs={'rel': 'uses', 'direction': 'out'}
            )
        self.generic_visit(node)

    # one-liner comment: Records standard import statements.
    def visit_Import(self, node: ast.Import):
        # 🚀 Add_edge with rel="uses" - classify directions (imports)
        for alias in node.names:
            self.lex.add_node({"nid": alias.name, "type": "PACKAGE_REF"})
            self.lex.add_edge(
                src=self.module_name, trgt=alias.name,
                attrs={'rel': 'uses', 'direction': 'import'}
            )
        self.generic_visit(node)

    # one-liner comment: Records 'from ... import ...' statements.
    def visit_ImportFrom(self, node: ast.ImportFrom):
        # 🚀 Add_edge with rel="uses" - classify directions (imports)
        source = node.module
        for alias in node.names:
            self.lex.add_edge(
                src=self.module_name, trgt=alias.name,
                attrs={'rel': 'uses', 'source_module': source}
            )
        self.generic_visit(node)

    # one-liner comment: Overrides function/async function visitors to apply custom processing.
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node)

    def visit_Name(self, node: ast.Name):
        self.generic_visit(node)


# --- Ray Actor Definitions (define each defined workflow inside a ray.remote) ---

@ray.remote
class RemoteRelay:
    """// Relay remote: Handles query processing and classification."""

    # one-liner comment: Initializes the RemoteRelay actor.
    def __init__(self, graph_engine):
        self.engine = graph_engine
        print("🚀 RemoteRelay initialized.")

    # one-liner comment: Processes the query, creates prompt versions, and classifies the intent.
    def process_query(self, query_str: str) -> Dict[str, Any]:
        # receive query string -> create 5 versions of prompt using a local llm (downloadable with pip) > classifier rlp
        # MOCK: creation/classification
        list_prompts = [f"{query_str} (V{i + 1})" for i in range(5)]
        classification = llm_call_sim(query_str, task="classify")

        print(f"🕵️ Query classified as: **{classification}**")
        return {"prompts": list_prompts, "classification": classification}


@ray.remote
class RemoteExpert:
    """// Expert remote: Handles external search, package installation, and knowledge base enrichment."""

    # one-liner comment: Initializes the RemoteExpert actor and defines local state for the Vector Store.
    def __init__(self, graph_engine):
        self.engine = graph_engine
        self.local_vs: List[Dict[str, Any]] = []  # define local vs
        self.static_search_prompt = "Gather information based specific instructions to support the underlying goal:"
        print("🧠 RemoteExpert initialized.")

    # one-liner comment: Fetches external data, installs packages (MOCK), and updates the Graph atomically.
    def fetch_data(self, list_prompts: List[str]):
        # rlp -> convert to 5 search queries -> gsearch request (MOCK)
        search_results = [{'content': f'Content with package_A for query {i + 1}', 'url': f'http://page{i + 1}.com'} for
                          i in range(5)]
        required_packages = {"mock_pkg_A"}  # MOCK extract: pip packages required for this task

        # run subprocess -> load packages in Graph
        for pkg in required_packages:
            self.engine.code_graph.add_node(
                {"nid": pkg, "type": "PACKAGE", "description": "Installed package."}
            )

        print(f"📦 Expert fetched data and logged packages: {required_packages}")


@ray.remote
class RemoteCollector:
    """// Collector remote: Handles similarity search, pathfinding, and code assembly."""

    # one-liner comment: Initializes the RemoteCollector.
    def __init__(self, graph_engine):
        self.engine = graph_engine
        print("🔍 RemoteCollector initialized.")

    # one-liner comment: Retrieves and assembles the final sorted and runnable code structure.
    def assemble_code(self, query_str: str) -> str:
        """// collect the sorted and runnable code structure incl all variables as str -> return"""

        # 1. Similarity Search (ss)
        related_nids = self.engine.cli_action_sem_search_and_path(query_str)
        collected_code_parts = {}

        # 2. Pathfinding and Collection
        # loop nids with sc > .9: include a pathfinding algirithmus...
        for nid in related_nids:
            # Note: Pathfinding logic is handled by the main engine class method
            pathway = self.engine.code_graph.get_neighbors_recursive(nid, relationship="needs")

            for path_nid in pathway:
                node_data = self.engine.code_graph.get_node(path_nid)
                if node_data and 'code' in node_data:
                    collected_code_parts[path_nid] = node_data['code']

        # 3. Collect parameters from .env (MOCK)
        env_vars = self.engine.env_vars

        # 4. Assemble and Return (Topologically Sorted - MOCK)
        assembled_code = f"# Assembled Code for Query: {query_str}\n"
        assembled_code += f"# ENV Vars: {env_vars}\n"
        assembled_code += "\n".join(collected_code_parts.values())
        assembled_code += "\n# --- End of Assembled Code ---"

        print(f"🧩 Code assembled from {len(collected_code_parts)} parts.")
        return assembled_code


@ray.remote
class RemoteExecutor:
    """// Executor remote: Executes the final codebase string."""

    # one-liner comment: Initializes the RemoteExecutor.
    def __init__(self, graph_engine):
        self.engine = graph_engine
        print("⚙️ RemoteExecutor initialized.")

    # one-liner comment: Executes the sorted code in an isolated environment and logs the result.
    # rcs: receive code str
    def execute_code(self, code_str: str) -> str:
        # rcs -> create runnable end executes the sorted codebase inside a ray.remote
        execution_log = self.engine.cli_action_run_remote(node_id="method_a",
                                                          input_params={})  # MOCK execution using a target node

        # ADD_CODE_GRAPH(with adapted code)
        self.engine.code_graph.add_node(
            {"nid": f"EXEC_LOG_{os.urandom(4).hex()}", "type": "EXECUTION_LOG", "code": code_str,
             "output": str(execution_log)}
        )
        return str(execution_log)


@ray.remote
class RemoteDebugger:
    """// Debugger remote: Handles monitoring and correction."""

    # one-liner comment: Initializes the RemoteDebugger.
    def __init__(self, graph_engine):
        self.engine = graph_engine
        print("🐛 RemoteDebugger initialized.")

    # one-liner comment: Executes workflow (MOCK), checks for traceback, and corrects via LLM.
    def debug_and_fix(self, workflow_name: str):
        # while loop runnable current code content entire Graph -> execute in subprocess each workflow

        # MOCK: Simulate execution error
        test_output = "Traceback:\nImportError: No module named 'missing_pkg'"
        if "ImportError" in test_output:
            # check traceback and debug wih a gem instance within a while loop

            # MOCK: Get faulty code (assuming it was the last thing run)
            faulty_code = "print(missing_pkg.hello)"

            debug_prompt = f"Fix CODE:\n{faulty_code}\nERROR:\n{test_output}"
            adapted_code = llm_call_sim(debug_prompt, task="debug_fix")

            # switch code in graph, keep prev code in "cache":str attr
            nid = "LAST_EXECUTED_CODE"  # Target node for correction
            prev_code = self.engine.code_graph.get_node(nid).get("code", "N/A")

            self.engine.code_graph.add_node(
                {"nid": nid, "type": "CODE_FIXED", "code": adapted_code, "cache": prev_code}
            )
            print(f"✅ Debugger fixed code and updated graph.")
            return "Debugger Status: Code Fixed."
        return "Debugger Status: No critical errors found."


@ray.remote
class RemoteCreator:
    """// Creator remote: Generates new code base."""

    # one-liner comment: Initializes the RemoteCreator.
    def __init__(self, graph_engine):
        self.engine = graph_engine
        print("✍️ RemoteCreator initialized.")

    # one-liner comment: Calls the LLM API to create new code and logs it to the graph.
    def create_code_base(self, prompt: str):
        # rlp -> gem api call: create code base
        new_code = llm_call_sim(prompt, task="create_code")

        # ADD_CODE_GRAPH
        self.engine.code_graph.add_node(
            {"nid": f"NEW_CODE_{os.urandom(4).hex()}", "type": "NEW_FILE", "code": new_code}
        )
        print(f"✅ Creator: New code generated.")


@ray.remote
class RemoteEditor:
    """// Editor remote: Modifies existing code."""

    # one-liner comment: Initializes the RemoteEditor.
    def __init__(self, graph_engine):
        self.engine = graph_engine
        print("✏️ RemoteEditor initialized.")

    # one-liner comment: Calls the LLM to perform changes on files and logs the change to the graph.
    def edit_code(self, prompt: str):
        # rcs -> llm call gem cli py client: static prompt: perform change on files
        edited_code = llm_call_sim(prompt, task="edit_code")

        # ADD_CODE_GRAPH
        self.engine.code_graph.add_node(
            {"nid": f"EDITED_CODE_{os.urandom(4).hex()}", "type": "EDITED_FILE", "code": edited_code}
        )
        print(f"✅ Editor: Code modified and change logged.")


# --- Main Engine Class ---

class CodeAnalyzerCLI:
    """// Manages the full workflow and executes CLI actions via Ray (Graph Engine Core)."""

    # one-liner comment: Initializes graph, actors, and loads base configuration.
    def __init__(self, codebase_root=CODEBASE_ROOT):
        print("✨ CodeAnalyzerCLI Initializing...")
        self.code_graph = LocalGraphUtils()
        self.codebase_root = codebase_root
        self.env_vars = load_env_variables(ENV_FILE_PATH)
        self.required_packages = load_requirements(REQUIREMENTS_FILE)

        # Initialize Remote Actors (each worker must be deployed from)
        self.relay_remote = RemoteRelay.options(name="RelayActor").remote(self)
        self.expert_remote = RemoteExpert.options(name="ExpertActor").remote(self)
        self.collector_remote = RemoteCollector.options(name="CollectorActor").remote(self)
        self.executor_remote = RemoteExecutor.options(name="ExecutorActor").remote(self)
        self.debugger_remote = RemoteDebugger.options(name="DebuggerActor").remote(self)
        self.creator_remote = RemoteCreator.options(name="CreatorActor").remote(self)
        self.editor_remote = RemoteEditor.options(name="EditorActor").remote(self)

        self._create_initial_files()  # Prepare environment
        self.create_nx_graph()  # Initial graph build
        self.cli_action_create_json()  # Create initial JSON dump
        print("✅ Engine setup complete.")

    # one-liner comment: Creates required placeholder files for robust testing.
    def _create_initial_files(self):
        """// Creates required placeholder files."""
        if not os.path.exists(REQUIREMENTS_FILE):
            with open(REQUIREMENTS_FILE, 'w') as f: f.write("numpy\nray\njax")
        if not os.path.exists(ENV_FILE_PATH):
            with open(ENV_FILE_PATH, 'w') as f: f.write("T=1.0\nMASS=0.1\nG_COUPLING=0.6")
        if not os.path.exists("example_module.py"):
            with open("example_module.py", 'w') as f:
                f.write("def method_a(x: float, y: jnp.ndarray): return x * jnp.sum(y)\n")
                self.code_graph.add_node(
                    {"nid": "LAST_EXECUTED_CODE", "type": "MOCK_CODE", "code": "def placeholder_func(): pass"})

    # one-liner comment: Builds the graph structure from the codebase, mapping files and modules.
    def create_nx_graph(self):
        """// Graph Engine: walk local dir -> add_node all folders type = MODULE..."""
        # MOCK: Simplified walk, relies on _create_initial_files
        file_name = "example_module.py"
        with open(file_name, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            inspector = StructInspector(self.code_graph, file_name, content)
            tree = ast.parse(content)
            inspector.visit(tree)
        except Exception as e:
            print(f"⚠️ Could not parse {file_name} during init: {e}")

        for package in self.required_packages:
            self.code_graph.add_node({"nid": package, "type": "PACKAGE"})

    # one-liner comment: Creates a JSON file dump of the current graph state.
    def cli_action_create_json(self, output_path=OUTPUT_GRAPH_JSON) -> str:
        """// Saves graph state using custom serialization."""
        try:
            # nx.link_node_data & pyvis render -> save files (json & html) in root
            graph_data = nx.node_link_data(self.code_graph.G)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, default=custom_serializer)
            print(f"✅ JSON file created.")
            return output_path
        except Exception as e:
            print(f"❌ Error serializing graph: {e}")
            return "ERROR"

    # one-liner comment: Performs SS and pathfinding to find related code and dependencies.
    def cli_action_sem_search_and_path(self, query: str) -> List[str]:
        """// Logic shared by Collector remote."""
        query_embedding = embed_text_sim(query)
        high_similarity_nodes: Dict[str, float] = {}

        for nid, data in self.code_graph.G.nodes(data=True):
            if 'embedding' in data:
                node_embedding = np.array(data['embedding'])
                score = get_similarity_score_sim(query_embedding, node_embedding)
                if score >= SIMILARITY_THRESHOLD:
                    high_similarity_nodes[nid] = score

        final_pathway_nodes = set()
        for nid, _ in high_similarity_nodes.items():
            path_nodes = self.code_graph.get_neighbors_recursive(nid, relationship="needs")
            final_pathway_nodes.update(path_nodes)

        return list(final_pathway_nodes)

    # one-liner comment: Executes the code remotely within a Ray worker.
    def cli_action_run_remote(self, node_id: str, input_params: Dict[str, Any] = {}):
        """// Wrapper for remote execution."""
        # This function definition is intentionally kept here as a wrapper to enable Ray actor execution
        # The execution logic is defined inside the Executor actor or the _execute_code_remote method.
        node_data = self.code_graph.get_node(node_id)
        if node_data and 'code' in node_data:
            # Execute remotely
            remote_task = RemoteExecutor._execute_code_remote.remote(self.executor_remote, node_data['code'],
                                                                     input_params, self.env_vars)
            return ray.get(remote_task)
        return {"Error": "Node or code not found for execution."}

    # --- UI/CLI (Final Implementation) ---

    # one-liner comment: Provides a terminal interface for user interaction and workflow initiation.
    def ui_terminal(self):
        """// terminal based ui to interact with the engine."""
        print("\n" + "=" * 60)
        print("🤖 Welcome, Benedikt, to the Graph Code Engine CLI!")
        print("=" * 60)

        while True:
            # whil loop and classifys all query inputs / blocks main thread
            print("\n**Available Options**:")
            print("1. Query Engine (Relay entry point)")
            print("2. Run Debugger Cycle")
            print("3. Exit")

            user_input = input("Enter option number or your query: ").strip()
            print("\n")

            if user_input.lower() in ['3', 'exit']:
                print("👋 Shutting down Ray and exiting.")
                ray.shutdown()
                break

            query = user_input if not user_input.isdigit() or user_input == '1' else None

            if query or user_input == '1':
                self._run_cli_action(query_str=query if query else input("Enter your query: ").strip(),
                                     action="RELAY_RUN")
            elif user_input == '2':
                self._run_cli_action(action="DEBUGGER_RUN")

    # one-liner comment: Executes the specific workflow based on the user's CLI action.
    def _run_cli_action(self, query_str: str = "", action: str = "RELAY_RUN"):
        """// define a clear step by step workflow that executes each possibel cli action."""

        if action == "RELAY_RUN":
            print("--- START WORKFLOW: RELAY_RUN ---")

            # 1. Relay remote
            relay_result = ray.get(self.relay_remote.process_query.remote(query_str))
            print(":: Next jump in local wf -> Relay finished.")

            # 2. Expert remote
            ray.get(self.expert_remote.fetch_data.remote(relay_result['prompts']))
            print(":: Next jump in local wf -> Expert finished.")

            # 3. Collector remote
            assembled_code = ray.get(self.collector_remote.assemble_code.remote(query_str))
            print(":: Next jump in local wf -> Collector finished. Code assembled.")

            # 4. Executor remote
            # MOCK Execution targeting the last assembled code structure
            execution_log = ray.get(self.executor_remote.execute_code.remote(assembled_code))
            print(f"Executor Result: {execution_log}")

            # 5. Save state
            self.cli_action_create_json()

        elif action == "DEBUGGER_RUN":
            print("--- START WORKFLOW: DEBUGGER_RUN ---")
            # 1. Debugger remote
            # while loop runnable current code content entire Graph -> execute in subprocess each workflow
            debugger_ref = self.debugger_remote.debug_and_fix.remote("MainWorkflow")
            debugger_log = ray.get(debugger_ref)
            print(f"🐞 {debugger_log}")
            self.cli_action_create_json()


# --- Execution Block ---

# one-liner comment: Defines and executes the step-by-step workflow for testing.
def main_call_altiems():
    """// main call altiems starts the cli."""
    print("🛠️ Starting system initialization...")
    cli = CodeAnalyzerCLI(codebase_root=CODEBASE_ROOT)
    cli.ui_terminal()


if __name__ == '__main__':
    main_call_altiems()


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


