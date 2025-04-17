import os
import openai
import logging
from dotenv import load_dotenv
from pathlib import Path
import weaviate
from weaviate.util import generate_uuid5
from weaviate.exceptions import ObjectAlreadyExistsException
import subprocess
import json
from tree_sitter import Language, Parser
import networkx as nx
from docx import Document  # For parsing .docx files
from nltk.tokenize import sent_tokenize
import nltk
import re
import weaviate.classes as wvc
from weaviate.connect import ConnectionParams
from weaviate.classes.query import Filter

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Add important files to track for special handling
KEY_FILES = {
    "strato-platform/strato/VM/SolidVM/solid-vm/src/Blockchain/SolidVM.hs": "core_vm",
    "strato-mercata-llm-docs/strato-platform/strato/indexer/slipstream/src/Slipstream/OutputData.hs": "indexer",
    "strato-mercata-llm-docs/strato-platform/strato/indexer/slipstream/src/Slipstream/Processor.hs": "indexer",
}

# Update SOURCE_DIRS to use tuples (directory_path, description)
SOURCE_DIRS = [
    # ("strato-platform/eth-bridge", "bridge related code"),
    # ("strato-platform/strato/api", "api related code"),
    # ("strato-platform/strato/indexer", "indexer related code"),
    # ("strato-platform/strato/VM", "VM code"),
    # ("strato-platform/marketplace", "marketplace related code"),
    # ("docs-website", "documentation website"),
    # ("strato-getting-started", "getting started guide"),
    # ("blockapps-rest", "rest api implementation for STRATO"),
    # ("smd-ui", "ui"),
    # ("strato-mercata-docs", "mercata documentation"),
    ("strato-platform/marketplace/backend/dapp/mercata-base-contracts", "contracts UTXO old"),
    ("strato-platform/mercata", "contracts ERC20"),
    # ("other", "Contains important .docx files")
]

# List of important onboarding documents
DOCS = [
    # ("Mercata.docx", "general"),
    # ("Mercata Integration ERC20.docx", "erc20"),
    # ("Mercata Integration UTXO.docx", "utxo")
]

# Add more extensions to capture all relevant files
EXTENSIONS = [".hs", ".sol", ".ts", ".tsx", ".md", ".js", ".jsx", ".json", ".yml", ".yaml", ".docx"]
CHUNK_SIZE = 50   # Reduced from 200 to 50 lines
OVERLAP = 15      # Slight adjustment from 20 to 15
DOC_CHUNK_SIZE = 1000  # Characters for document chunks
DOC_OVERLAP = 250  # Characters of overlap for document chunks

# Update client initialization for Weaviate v4 with timeout
client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=8080,
    http_secure=False,
    grpc_host="localhost",
    grpc_port=50051,
    grpc_secure=False,
    skip_init_checks=True
)

# Set up Tree-sitter
def setup_tree_sitter():
    """Initialize Tree-sitter for Haskell, Solidity, and JavaScript parsing."""
    logger.info(f"CWD is {os.getcwd()}, entries here: {os.listdir()}")
    logger.info("Setting up Tree-sitter parsers")
    
    # Check if tree-sitter libraries exist, if not, build them
    if not os.path.exists("./tree-sitter-haskell") or not os.path.exists("./tree-sitter-solidity") or not os.path.exists("./tree-sitter-javascript"):
        logger.info("Tree-sitter grammars not found, installing...")
        
        # Clone grammar repositories if they don't exist
        if not os.path.exists("./tree-sitter-haskell"):
            subprocess.run(["git", "clone", "https://github.com/tree-sitter/tree-sitter-haskell.git"])
        
        if not os.path.exists("./tree-sitter-solidity"):
            subprocess.run(["git", "clone", "https://github.com/JoranHonig/tree-sitter-solidity.git"])
            
        if not os.path.exists("./tree-sitter-javascript"):
            subprocess.run(["git", "clone", "https://github.com/tree-sitter/tree-sitter-javascript.git"])
    
    # Create build directory if it doesn't exist
    os.makedirs("build", exist_ok=True)
    
    # Build the language library properly using Language.build_library
    try:
        logger.info("Building Tree-sitter language library...")
        Language.build_library(
            'build/languages.so',
            [
                './tree-sitter-haskell',
                './tree-sitter-solidity',
                './tree-sitter-javascript'
            ]
        )
        logger.info("🌳 Successfully built Tree-sitter language library")
    except Exception as e:
        logger.exception(f"❌ Error building Tree-sitter language library: {e}")
        # Don't silently return an empty dict, let the error propagate up
        raise
    
    # Load the languages from the built library
    try:
        HASKELL = Language('build/languages.so', 'haskell')
        SOLIDITY = Language('build/languages.so', 'solidity')
        JAVASCRIPT = Language('build/languages.so', 'javascript')
        
        haskell_parser = Parser()
        haskell_parser.set_language(HASKELL)
        
        solidity_parser = Parser()
        solidity_parser.set_language(SOLIDITY)
        
        javascript_parser = Parser()
        javascript_parser.set_language(JAVASCRIPT)
        
        parsers = {
            ".hs": haskell_parser,
            ".sol": solidity_parser,
            ".js": javascript_parser,
            ".jsx": javascript_parser  # Use same parser for JSX
        }
        
        logger.info(f"Loaded parsers for extensions: {list(parsers.keys())}")
        return parsers
    except Exception as e:
        logger.exception(f"❌ Error loading tree-sitter languages: {e}")
        # Don't return empty dict, fail visibly
        raise

# Graph extraction functions
def extract_haskell_definitions(node, file_path, source_code, graph):
    """Extract function and type definitions from Haskell code."""
    definitions = []
    
    def traverse(node, parent_name=None):
        # Update node types to match actual Haskell Tree-sitter grammar
        if node.type == "value_declaration":
            # Extract the function name
            name_node = None
            for child in node.children:
                if child.type == "variable" or child.type == "function":
                    name_node = child
                    break
            
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode('utf8')
                definitions.append({
                    "type": "function",
                    "name": name,
                    "file": str(file_path),
                    "line": node.start_point[0] + 1
                })
                
                # Add to graph
                graph.add_node(name, type="function", file=str(file_path))
                if parent_name:
                    graph.add_edge(parent_name, name)
        
        # Match types: data declarations, type aliases, and newtypes
        elif node.type in ["data_declaration", "type_synonym_declaration", "newtype_declaration"]:
            name_node = None
            for child in node.children:
                if child.type == "type_constructor":
                    name_node = child
                    break
            
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode('utf8')
                definitions.append({
                    "type": "type",
                    "name": name,
                    "file": str(file_path),
                    "line": node.start_point[0] + 1
                })
                
                # Add to graph
                graph.add_node(name, type="type", file=str(file_path))
        
        # Add function call extraction - look for variable references
        elif node.type == "exp_apply":
            # In Haskell, an application might be a function call
            for child in node.children:
                if child.type == "variable" and parent_name:
                    called_func = source_code[child.start_byte:child.end_byte].decode("utf8")
                    if called_func != parent_name:
                        graph.add_edge(parent_name, called_func, inferred=True)
                
        # Recurse through children
        for child in node.children:
            traverse(child, parent_name)
    
    traverse(node)
    return definitions

def extract_solidity_definitions(node, file_path, source_code, graph):
    """Extract function, event, and contract definitions from Solidity code."""
    definitions = []
    
    def extract_callee(n, source_code):
        """Extract the actual identifier or property from member calls."""
        if n.type == "identifier":
            return source_code[n.start_byte:n.end_byte].decode('utf8')
        elif n.type == "member_expression":
            for child in reversed(n.children):
                if child.type in {"identifier", "property_identifier"}:
                    return source_code[child.start_byte:child.end_byte].decode('utf8')
        elif n.type == "call_expression":
            fn_node = n.child_by_field_name("function")
            if fn_node:
                return extract_callee(fn_node, source_code)
        elif n.type == "type_cast_expression":
            # Handle cases like Sale(sale).autoTransfer(...)
            member_expr = next((c for c in n.children if c.type == "member_expression"), None)
            if member_expr:
                return extract_callee(member_expr, source_code)
        elif n.child_count > 0:
            for c in reversed(n.children):
                result = extract_callee(c, source_code)
                if result:
                    return result
        return None
    
    def traverse(node, parent_contract=None, parent_name=None):
        if node.type == "contract_definition":
            contract_name = None
            for child in node.children:
                if child.type == "identifier":
                    contract_name = source_code[child.start_byte:child.end_byte].decode('utf8')
                    break
            
            if contract_name:
                definitions.append({
                    "type": "contract",
                    "name": contract_name,
                    "file": str(file_path),
                    "line": node.start_point[0] + 1
                })
                
                # Add to graph
                graph.add_node(contract_name, type="contract", file=str(file_path))
                
                # Process contract body with this contract as parent
                for child in node.children:
                    traverse(child, contract_name, contract_name)
        
        # Function definitions
        elif node.type == "function_definition":
            function_name = None
            visibility = "default"
            state_mutability = "non-payable"
            
            for child in node.children:
                if child.type == "function_head":
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            function_name = source_code[subchild.start_byte:subchild.end_byte].decode('utf8')
                        elif subchild.type in ["visibility", "mutability"]:
                            prop_value = source_code[subchild.start_byte:subchild.end_byte].decode('utf8')
                            if prop_value in ["public", "private", "internal", "external"]:
                                visibility = prop_value
                            elif prop_value in ["view", "pure", "payable"]:
                                state_mutability = prop_value
            
            if function_name and parent_contract:
                # Create function name with contract prefix for better relationship tracking
                full_name = f"{parent_contract}::{function_name}"
                
                definitions.append({
                    "type": "function",
                    "name": full_name,
                    "short_name": function_name,
                    "file": str(file_path),
                    "line": node.start_point[0] + 1,
                    "visibility": visibility,
                    "state_mutability": state_mutability
                })
                
                # Add to graph with improved node attributes
                graph.add_node(full_name, 
                               type="function", 
                               file=str(file_path), 
                               contract=parent_contract,
                               function_name=function_name,
                               visibility=visibility)
                
                if parent_contract:
                    graph.add_edge(parent_contract, full_name, type="contains")
                
                # Process function body for calls
                for child in node.children:
                    if child.type == "function_body":
                        traverse(child, parent_contract, full_name)
                    elif child.type == "function_head":
                        # Check for inheritance and overrides
                        for subchild in child.children:
                            if subchild.type == "override_specifier":
                                graph.nodes[full_name]["overrides"] = True
        
        # Function call extraction with improved extraction
        elif node.type in ["call_expression", "function_call_expression"]:
            if parent_name:
                callee = extract_callee(node, source_code)
                
                # Skip Solidity built-in functions
                solidity_builtins = {'require', 'assert', 'revert', 'selfdestruct', 'gasleft'}
                if callee and callee not in solidity_builtins:
                    # For now just store the function name, we'll resolve contract later
                    graph.add_edge(parent_name, callee, type="calls", inferred=True)
                    logger.debug(f"Found call: {parent_name} → {callee}")
        
        # Emit event statement
        elif node.type == "emit_statement":
            if parent_name:
                for child in node.children:
                    if child.type == "function_call_expression":
                        for subchild in child.children:
                            if subchild.type == "identifier":
                                emitted_event = source_code[subchild.start_byte:subchild.end_byte].decode('utf8')
                                full_event_name = f"{parent_contract}.{emitted_event}" if parent_contract else emitted_event
                                graph.add_edge(parent_name, full_event_name, type="emits", inferred=True)
                                break
        
        # Recurse through children
        for child in node.children:
            traverse(child, parent_contract, parent_name)
    
    traverse(node)
    
    # Post-processing: Resolve function calls across contracts
    function_to_contract = {}
    # First build a mapping of function short names to their full names with contracts
    for node, attrs in graph.nodes(data=True):
        if attrs.get("type") == "function" and "function_name" in attrs:
            function_name = attrs["function_name"]
            function_to_contract[function_name] = node
    
    # Now resolve the edges that only have function names
    resolved_edges = []
    for source, target, data in list(graph.edges(data=True)):
        if data.get("inferred") and "::" not in target:
            # Try to find which contract this function belongs to
            if target in function_to_contract:
                full_target = function_to_contract[target]
                # Remove the old edge and add the resolved one
                graph.remove_edge(source, target)
                graph.add_edge(source, full_target, type="calls")
                resolved_edges.append((source, full_target))
    
    logger.debug(f"Resolved {len(resolved_edges)} function call edges in Solidity")
    return definitions

# Add JavaScript definition extraction function
def extract_javascript_definitions(node, file_path, source_code, graph):
    """Extract function, class, and variable definitions from JavaScript code."""
    definitions = []
    
    def traverse(node, parent_name=None):
        # Update node types to match JavaScript Tree-sitter grammar
        if node.type == "function_declaration" or node.type == "method_definition":
            name_node = None
            
            # Function declarations have a name child
            if node.type == "function_declaration":
                for child in node.children:
                    if child.type == "identifier":
                        name_node = child
                        break
            # Method definitions have a name property
            elif node.type == "method_definition":
                for child in node.children:
                    if child.type == "property_identifier":
                        name_node = child
                        break
            
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode('utf8')
                full_name = f"{parent_name}.{name}" if parent_name else name
                
                definitions.append({
                    "type": "function",
                    "name": full_name,
                    "file": str(file_path),
                    "line": node.start_point[0] + 1
                })
                
                # Add to graph
                graph.add_node(full_name, type="function", file=str(file_path))
                if parent_name:
                    graph.add_edge(parent_name, full_name)
                
                # Process function body with this function as parent
                for child in node.children:
                    if child.type == "statement_block":
                        traverse(child, full_name)
        
        # Extract class definitions
        elif node.type == "class_declaration":
            name_node = None
            for child in node.children:
                if child.type == "identifier":
                    name_node = child
                    break
            
            if name_node:
                name = source_code[name_node.start_byte:name_node.end_byte].decode('utf8')
                definitions.append({
                    "type": "class",
                    "name": name,
                    "file": str(file_path),
                    "line": node.start_point[0] + 1
                })
                
                # Add to graph
                graph.add_node(name, type="class", file=str(file_path))
                
                # Process class body with this class as parent
                for child in node.children:
                    if child.type == "class_body":
                        traverse(child, name)
        
        # Extract variable declarations with arrow functions
        elif node.type in ["lexical_declaration", "variable_declaration"]:
            for child in node.children:
                if child.type == "variable_declarator":
                    name_node = None
                    is_function = False
                    
                    for subchild in child.children:
                        if subchild.type == "identifier":
                            name_node = subchild
                        elif subchild.type == "arrow_function":
                            is_function = True
                    
                    if name_node and is_function:
                        name = source_code[name_node.start_byte:name_node.end_byte].decode('utf8')
                        full_name = f"{parent_name}.{name}" if parent_name else name
                        
                        definitions.append({
                            "type": "function",
                            "name": full_name,
                            "file": str(file_path),
                            "line": node.start_point[0] + 1
                        })
                        
                        # Add to graph
                        graph.add_node(full_name, type="function", file=str(file_path))
                        if parent_name:
                            graph.add_edge(parent_name, full_name)
        
        # Extract React components in JavaScript
        elif node.type == "export_statement":
            for child in node.children:
                if child.type == "object":
                    # Default name with line number if we can't determine component name
                    component_name = f"Component_{node.start_point[0]}"
                    
                    # Find if this is assigned to a variable
                    prev_node = node.prev_sibling
                    if prev_node and prev_node.type in ["lexical_declaration", "variable_declaration"]:
                        for subchild in prev_node.children:
                            if subchild.type == "variable_declarator":
                                for var_child in subchild.children:
                                    if var_child.type == "identifier":
                                        component_name = source_code[var_child.start_byte:var_child.end_byte].decode('utf8')
                                        break
                    
                    # Add component to definitions
                    definitions.append({
                        "type": "component",
                        "name": component_name,
                        "file": str(file_path),
                        "line": node.start_point[0] + 1
                    })
                    
                    # Add to graph
                    graph.add_node(component_name, type="component", file=str(file_path))
                    
                    # Extract method definitions from the component object
                    traverse(child, component_name)
        
        # Extract function calls to build the call graph
        elif node.type == "call_expression":
            if parent_name:
                function_name_node = node.child_by_field_name("function")
                if function_name_node:
                    function_name = source_code[function_name_node.start_byte:function_name_node.end_byte].decode('utf8')
                    # Add edge showing function call relationship
                    if function_name != parent_name:  # Avoid self-calls
                        graph.add_edge(parent_name, function_name, type="calls", inferred=True)
        
        # Recurse for all child nodes
        for child in node.children:
            traverse(child, parent_name)
    
    traverse(node)
    return definitions

# Parse directory comments to create a mapping of directories to their purpose
def extract_directory_metadata():
    """Extract metadata from SOURCE_DIRS tuples to enhance the graph."""
    directory_metadata = {}
    
    for directory_path, description in SOURCE_DIRS:
        # Store the description and extract first word as purpose
        purpose = description.split()[0] if description else "unknown"
        directory_metadata[directory_path] = {
            "description": description,
            "purpose": purpose
        }
    
    logger.info(f"Extracted metadata for {len(directory_metadata)} directories")
    return directory_metadata

# Modify extract_code_graph to use directory metadata
def extract_code_graph(parsers):
    """Extract a graph of code relationships across the codebase."""
    logger.info("Extracting code graph from codebase")
    
    # Add diagnostic logging to see what file types exist vs what parsers we have
    logger.info(f"Parser extensions available: {list(parsers.keys())}")
    
    # Scan the directories for all file suffixes to see what's actually there
    all_suffixes = set()
    for directory_path, _ in SOURCE_DIRS:
        source_path = Path(directory_path)
        if not source_path.exists():
            continue
        
        for root, _, files in os.walk(source_path):
            for f in files:
                path = Path(root) / f
                all_suffixes.add(path.suffix.lower())  # Normalize to lowercase
                
    logger.info(f"All file suffixes found in directories: {sorted(all_suffixes)}")
    logger.info(f"Parsers will be used for: {sorted(set(all_suffixes) & set(parsers.keys()))}")
    
    graph = nx.DiGraph()
    definitions = []
    
    # Extract directory metadata from tuples
    dir_metadata = extract_directory_metadata()
    
    for directory_path, _ in SOURCE_DIRS:
        source_path = Path(directory_path)
        if not source_path.exists():
            continue
            
        # Get directory purpose from metadata
        dir_purpose = "unknown"
        if directory_path in dir_metadata:
            dir_purpose = dir_metadata[directory_path].get("purpose", "unknown")
        
        for root, _, files in os.walk(source_path):
            for f in files:
                path = Path(root) / f
                if path.suffix in [".hs", ".sol", ".js", ".jsx"]:
                    logger.debug(f"Parsing {path}")
                    
                    # Check if we have a parser for this file type
                    if path.suffix not in parsers:
                        continue
                        
                    try:
                        with open(path, 'rb') as file:
                            source_code = file.read()
                            
                        parser = parsers[path.suffix]
                        tree = parser.parse(source_code)
                        
                        # DEBUG: Print top-level node types
                        if logger.level <= logging.DEBUG:
                            logger.debug(f"AST for {path.name}:")
                            for child in tree.root_node.children:
                                logger.debug(f"  Node type: {child.type}")
                        
                        # Extract definitions based on file type
                        if path.suffix == ".hs":
                            file_defs = extract_haskell_definitions(tree.root_node, path, source_code, graph)
                        elif path.suffix == ".sol":
                            file_defs = extract_solidity_definitions(tree.root_node, path, source_code, graph)
                        elif path.suffix in [".js", ".jsx"]:
                            file_defs = extract_javascript_definitions(tree.root_node, path, source_code, graph)
                        else:
                            file_defs = []
                            
                        # Add directory purpose to each definition
                        for def_item in file_defs:
                            def_item["directory_purpose"] = dir_purpose
                            
                            # Update graph nodes with purpose
                            if "name" in def_item:
                                if def_item["name"] in graph.nodes:
                                    graph.nodes[def_item["name"]]["directory_purpose"] = dir_purpose
                        
                        definitions.extend(file_defs)
                        
                        # Special handling for key files
                        str_path = str(path)
                        if str_path in KEY_FILES:
                            for def_item in file_defs:
                                def_item["importance"] = "high"
                                def_item["subsystem"] = KEY_FILES[str_path]
                                
                                # Update graph node attributes
                                if def_item["type"] == "function" and "name" in def_item:
                                    graph.nodes[def_item["name"]]["importance"] = "high"
                                    graph.nodes[def_item["name"]]["subsystem"] = KEY_FILES[str_path]
                        
                        # Special handling for marketplace JavaScript files
                        if "strato-platform/marketplace" in str_path and path.suffix in [".js", ".jsx"]:
                            for def_item in file_defs:
                                def_item["marketplace"] = True
                                def_item["importance"] = "medium"  # Make marketplace components visible in the graph
                                
                                # Update graph node attributes
                                if "name" in def_item:
                                    graph.nodes[def_item["name"]]["marketplace"] = True
                                    graph.nodes[def_item["name"]]["importance"] = "medium"
                    
                    except Exception as e:
                        logger.error(f"Error parsing {path}: {e}")
    
    # Add category labels from classify_path
    for node in graph.nodes:
        node_data = graph.nodes[node]
        if "file" in node_data:
            classification = classify_path(Path(node_data["file"]))
            graph.nodes[node].update(classification)
    
    # Add semantic edges based on directory purposes
    add_semantic_edges(graph, dir_metadata)
    
    # Save the graph as a JSON file
    graph_data = nx.node_link_data(graph, edges="links")
    with open("code_graph.json", "w") as f:
        json.dump(graph_data, f, indent=2)
    
    return definitions, graph

def add_semantic_edges(graph, dir_metadata):
    """Add semantic edges based on directory purposes."""
    # Group nodes by their directory purpose
    purpose_groups = {}
    for node, data in graph.nodes(data=True):
        purpose = data.get("directory_purpose", "unknown")
        if purpose not in purpose_groups:
            purpose_groups[purpose] = []
        purpose_groups[purpose].append(node)
    
    # Create "related_to" edges between components with specific relationships
    related_purposes = {
        ("bridge", "api"): "connects_to",
        ("indexer", "api"): "provides_data_to",
        ("vm", "indexer"): "emits_events_to",
        ("contract", "marketplace"): "used_by",
        ("documentation", "api"): "describes",
        ("documentation", "contract"): "describes",
        ("ui", "api"): "consumes"
    }
    
    edge_count = 0
    for (purpose1, purpose2), relation_type in related_purposes.items():
        if purpose1 in purpose_groups and purpose2 in purpose_groups:
            # Add relationships between important nodes from different components
            for node1 in purpose_groups[purpose1]:
                for node2 in purpose_groups[purpose2]:
                    # Only connect important nodes to avoid too many edges
                    if graph.nodes[node1].get("importance") == "high" or graph.nodes[node2].get("importance") == "high":
                        if not graph.has_edge(node1, node2):
                            graph.add_edge(node1, node2, type=relation_type, semantic=True)
                            edge_count += 1
    
    logger.info(f"Added {edge_count} semantic edges based on directory purposes")
    
    # Identify imported files and add import relationships
    add_import_relationships(graph)
    
    # Add specific code-documentation links through keyword matching
    connect_code_to_documentation(graph)

def add_import_relationships(graph):
    """Add edges representing import relationships between files."""
    logger.info("Analyzing import relationships between files...")
    
    # Group nodes by file
    file_to_nodes = {}
    for node, data in graph.nodes(data=True):
        if "file" in data:
            file_path = data["file"]
            if file_path not in file_to_nodes:
                file_to_nodes[file_path] = []
            file_to_nodes[file_path].append(node)
    
    # Track import relationships we've found
    imports_found = 0
    
    # For each file with solidity nodes
    for file_path, nodes in file_to_nodes.items():
        if not file_path.endswith(".sol"):
            continue
        
        try:
            # Read the file to look for imports
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
            # Simple regex to find imports
            import_pattern = r'import\s+["\'](.+?)["\']'
            import_matches = re.findall(import_pattern, file_content)
            
            for imported_path in import_matches:
                # Resolve the imported path relative to current file
                dir_path = os.path.dirname(file_path)
                if imported_path.startswith('./'):
                    resolved_path = os.path.normpath(os.path.join(dir_path, imported_path))
                else:
                    # This is a simplified approach - real resolution is more complex
                    resolved_path = os.path.join(dir_path, imported_path)
                
                # Find matching file in our nodes
                potential_matches = [f for f in file_to_nodes.keys() 
                                    if os.path.basename(f) == os.path.basename(resolved_path)]
                
                if potential_matches:
                    # For each node in importing file, add relationship to imports
                    for importer_node in nodes:
                        # Only add import edges for contract nodes
                        if graph.nodes[importer_node].get("type") == "contract":
                            for target_file in potential_matches:
                                for target_node in file_to_nodes[target_file]:
                                    if graph.nodes[target_node].get("type") == "contract":
                                        if not graph.has_edge(importer_node, target_node):
                                            graph.add_edge(importer_node, target_node, 
                                                          type="imports", file_import=True)
                                            imports_found += 1
        except Exception as e:
            logger.warning(f"Error processing imports for {file_path}: {e}")
    
    logger.info(f"Added {imports_found} import relationships between contracts")

def connect_code_to_documentation(graph):
    """Connect code entities with related documentation based on keyword matching."""
    logger.info("Creating relationships between code and documentation...")
    
    # Identify documentation nodes
    doc_nodes = []
    code_nodes = []
    for node, data in graph.nodes(data=True):
        if data.get("category") == "docs" or "documentation" in str(data.get("directory_purpose", "")):
            doc_nodes.append(node)
        elif data.get("type") in ["function", "contract", "class", "component"]:
            code_nodes.append((node, data))
    
    logger.info(f"Found {len(doc_nodes)} documentation nodes and {len(code_nodes)} code nodes")
    
    # No documentation found, exit early
    if not doc_nodes:
        logger.warning("No documentation nodes found in the graph")
        return
    
    # Extract keywords from code nodes (function names, contract names, etc.)
    code_keywords = {}
    for node, data in code_nodes:
        # Extract the short name (without parent prefix)
        name = node.split(".")[-1] if "." in node else node
        
        # Skip very short names (less than 4 chars) which could cause false matches
        if len(name) < 4:
            continue
            
        # Add to keywords dict with original node
        code_keywords[name.lower()] = node
        
        # Also add variations for common naming patterns
        if "contract" in name.lower():
            simplified = name.lower().replace("contract", "")
            if len(simplified) >= 4:
                code_keywords[simplified] = node
                
        # Handle CamelCase by adding lower_case_with_underscores version
        if any(c.isupper() for c in name[1:]):
            snake_case = ''.join(['_'+c.lower() if c.isupper() else c.lower() for c in name]).lstrip('_')
            if len(snake_case) >= 4:
                code_keywords[snake_case] = node
    
    # Now search for these keywords in documentation nodes
    relationships_created = 0
    
    for doc_node in doc_nodes:
        doc_data = graph.nodes[doc_node]
        doc_text = str(doc_node).lower()  # Use node name as minimal text
        
        # If there's a file attribute, extract it for better matching
        doc_file = str(doc_data.get("file", "")).lower()
        
        # Combine all text fields for searching
        search_text = f"{doc_text} {doc_file}"
        
        # Find code keywords in documentation text
        for keyword, code_node in code_keywords.items():
            if keyword in search_text:
                # Create a relationship
                if not graph.has_edge(doc_node, code_node):
                    graph.add_edge(doc_node, code_node, type="documents", semantic=True)
                    relationships_created += 1
                    
                    # Also create reverse relationship for easier navigation
                    if not graph.has_edge(code_node, doc_node):
                        graph.add_edge(code_node, doc_node, type="documented_by", semantic=True)
                        relationships_created += 1
    
    logger.info(f"Created {relationships_created} relationships between code and documentation")

def classify_path(path: Path) -> dict:
    path_str = str(path)
    classification = {"category": "unknown"}
    
    # Primary classification
    if "solid-vm" in path_str:
        classification["category"] = "vm"
        classification["module"] = "solidvm" if "SolidVM.hs" in path_str else "solidvmspec"
    elif "Typechecker" in path_str:
        classification["category"] = "vm"
        classification["module"] = "typechecker"
    elif "cirrus" in path_str.lower():
        classification["category"] = "indexer"
        classification["module"] = "cirrus"
    elif "slipstream" in path_str:
        classification["category"] = "indexer"
        classification["module"] = "slipstream"
    elif "mercata-base-contracts" in path_str:
        classification["category"] = "contract"
        classification["model"] = "utxo"
    elif "mercata" in path_str:
        classification["category"] = "contract"
        classification["model"] = "erc20"
    elif "smd-ui" in path_str:
        classification["category"] = "ui"
        classification["module"] = "smd-explorer"
    elif "strato-mercata-docs" in path_str:
        classification["category"] = "docs"
        classification["module"] = "smd-documentation"
    elif "docs-website" in path_str:
        classification["category"] = "docs"
        classification["module"] = "strato-docs" if "cirrus" in path_str.lower() else "general-docs"
    elif "strato-api-tests" in path_str:
        classification["category"] = "api-tests"
    elif "strato-getting-started" in path_str:
        classification["category"] = "setup"
    elif "blockapps-rest" in path_str:
        classification["category"] = "rest-api"
        classification["module"] = "strato-api"
        classification["usecase"] = "blockchain_interaction"
    elif "other" in path_str:
        classification["category"] = "misc"
    
    # Add Cirrus-specific classification for docs
    if "cirrus" in path_str.lower():
        classification["subsystem"] = "cirrus"
        classification["usecase"] = "data_indexing"
    
    return classification

def init_schema():
    logger.info("Initializing Weaviate schema")
    
    # Get existing collection names
    collections = client.collections.list_all()
    # Fix: collections is already a list of strings in Weaviate v4
    existing_collections = collections
    
    # Create CodeChunk collection
    if "CodeChunk" not in existing_collections:
        code_chunk = client.collections.create(
            name="CodeChunk",
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_transformers(),
            description="Code chunks from Strato PBFT L1 ecosystem",
            properties=[
                weaviate.classes.config.Property(name="text", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="file", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="category", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="module", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="model", data_type=weaviate.classes.config.DataType.TEXT)
            ]
        )
        logger.info("✅ Created 'CodeChunk' collection")
    else:
        logger.info("ℹ️ 'CodeChunk' collection already exists")

    # Create CodeDefinition collection
    if "CodeDefinition" not in existing_collections:
        client.collections.create(
            name="CodeDefinition",
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_transformers(),
            description="Code definitions (functions, types, contracts) from the codebase",
            properties=[
                weaviate.classes.config.Property(name="name", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="type", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="file", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="line", data_type=weaviate.classes.config.DataType.INT),
                weaviate.classes.config.Property(name="category", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="module", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="model", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="importance", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="subsystem", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="visibility", data_type=weaviate.classes.config.DataType.TEXT)
            ]
        )
        logger.info("✅ Created 'CodeDefinition' collection")
    else:
        logger.info("ℹ️ 'CodeDefinition' collection already exists")
    
    # Create GraphRelationship collection
    if "GraphRelationship" not in existing_collections:
        client.collections.create(
            name="GraphRelationship",
            description="Relationships between code definitions",
            properties=[
                weaviate.classes.config.Property(name="source", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="target", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="type", data_type=weaviate.classes.config.DataType.TEXT)
            ]
        )
        logger.info("✅ Created 'GraphRelationship' collection")
    else:
        logger.info("ℹ️ 'GraphRelationship' collection already exists")
        
    # Create MercataOnboardingDoc collection
    if "MercataOnboardingDoc" not in existing_collections:
        client.collections.create(
            name="MercataOnboardingDoc",
            vectorizer_config=weaviate.classes.config.Configure.Vectorizer.text2vec_transformers(),
            description="Onboarding and listing guides for ERC20/UTXO models with Mercata",
            properties=[
                weaviate.classes.config.Property(name="text", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="source", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="model", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="section", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="usecase", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="heading", data_type=weaviate.classes.config.DataType.TEXT),
                weaviate.classes.config.Property(name="chunk_id", data_type=weaviate.classes.config.DataType.INT)
            ]
        )
        logger.info("✅ Created 'MercataOnboardingDoc' collection")
    else:
        logger.info("ℹ️ 'MercataOnboardingDoc' collection already exists")
        

def find_file_in_codebase(file_pattern, limit=5):
    """Find specific files in the codebase with robust matching and debugging."""
    # Get collection
    collection = client.collections.get("CodeChunk")
    
    # Log search attempt for debugging
    logger.info(f"Searching for file pattern: '{file_pattern}'")
    
    # Fetch all code chunks to examine what's in the database
    all_results = collection.query.fetch_objects(
        limit=100,  # Get enough documents to find patterns
        return_properties=["text", "file", "category", "module"]
    ).objects
    
    # Debug: Log some sample file paths to see what's in the database
    sample_files = set()
    for obj in all_results[:20]:  # Look at first 20 results
        if "file" in obj.properties:
            sample_files.add(obj.properties["file"])
    
    logger.info(f"Sample files in database: {list(sample_files)[:10]}")
    
    # Try multiple search strategies
    matching_results = []
    
    # STRATEGY 1: Direct case-insensitive search
    for obj in all_results:
        if "file" in obj.properties:
            file_path = obj.properties["file"]
            # Check if our pattern appears anywhere in the path (case-insensitive)
            if file_pattern.lower() in file_path.lower():
                matching_results.append(obj)
                logger.info(f"MATCH FOUND: '{file_pattern}' in '{file_path}'")
    
    # STRATEGY 2: Look for just the filename part
    if not matching_results:
        for obj in all_results:
            if "file" in obj.properties:
                file_path = obj.properties["file"]
                # Extract just the filename from the path
                if "/" in file_path:
                    filename = file_path.split("/")[-1]
                else:
                    filename = file_path
                
                # Compare with our search pattern
                if file_pattern.lower() == filename.lower():
                    matching_results.append(obj)
                    logger.info(f"FILENAME MATCH: '{file_pattern}' matches '{filename}' in '{file_path}'")
    
    # STRATEGY 3: Special handling for known files
    if not matching_results and file_pattern.lower() == "escrow.sol":
        # Try brute force search for specific keywords in text content
        for obj in all_results:
            if "text" in obj.properties and "file" in obj.properties:
                if "escrow" in obj.properties["text"].lower():
                    matching_results.append(obj)
                    logger.info(f"CONTENT MATCH: 'escrow' in text content of '{obj.properties['file']}'")
    
    # Log results
    if matching_results:
        logger.info(f"Found {len(matching_results)} matches for '{file_pattern}'")
    else:
        logger.warning(f"No matches found for '{file_pattern}' after trying multiple strategies")
    
    return matching_results[:limit]

# Modify chunk_file to include file metadata in the chunk text
def chunk_file(path: Path):
    """
    Create chunks from a file, including file metadata in the text to improve searchability.
    Returns iterator of (chunk_text, start_line, end_line) tuples.
    """
    try:
        # First try with utf-8
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            # If utf-8 fails, try with latin-1 which accepts any byte value
            with open(path, "r", encoding="latin-1") as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return []
    
    # Get file extension for better context
    file_ext = path.suffix.lstrip('.')
    file_name = path.name
    
    # Add more detailed header with file info to improve semantic search
    file_header = f"// File: {path} ({file_ext})\n// Name: {file_name}\n// Description: Code from {path.parent}\n\n"
    
    # For very large files, create even smaller chunks to improve search
    actual_chunk_size = min(CHUNK_SIZE, max(30, len(lines) // 10)) if len(lines) > 500 else CHUNK_SIZE
    
    for i in range(0, len(lines), actual_chunk_size - OVERLAP):
        chunk_lines = lines[i:i + actual_chunk_size]
        # Include line numbers in the header for better context
        chunk_header = f"{file_header}// Lines: {i+1} to {i+len(chunk_lines)}\n\n"
        # Include file metadata at the beginning of each chunk
        chunk_text = chunk_header + "".join(chunk_lines)
        yield chunk_text, i, i + len(chunk_lines)

def scan_and_upload():
    logger.info(f"Starting code scan and upload from directories: {[path for path, _ in SOURCE_DIRS]}")
    total_files = 0
    total_chunks = 0
    file_metadata = {}

    # Verify Weaviate connection first
    try:
        logger.info("Testing Weaviate connection...")
        collection = client.collections.get("CodeChunk")
        logger.info("✅ Weaviate connection successful")
    except Exception as e:
        logger.error(f"⚠️ Weaviate connection error: {e}")
        raise e
    
    for directory_path, _ in SOURCE_DIRS:
        # Skip 'other' directory if it doesn't exist to avoid unnecessary warnings
        if directory_path == "other" and not Path(directory_path).exists():
            continue
            
        source_path = Path(directory_path)
        if not source_path.exists():
            logger.warning(f"Source directory {directory_path} does not exist, skipping.")
            continue
            
        logger.info(f"Processing directory: {directory_path}")
        
        # Count files first to know what we're working with
        files_to_process = []
        for root, _, files in os.walk(source_path):
            for f in files:
                path = Path(root) / f
                if path.suffix in EXTENSIONS:
                    files_to_process.append(path)
        
        logger.info(f"Found {len(files_to_process)} files to process in {directory_path}")
        
        # Now process the files
        for path in files_to_process:
            meta = classify_path(path)
            logger.info(f"Processing file: {path}")
            total_files += 1
            chunk_count = 0
            
            # Store basic metadata
            file_metadata[str(path)] = {
                "category": meta.get("category", "unknown"),
                "module": meta.get("module", "")
            }
            
            # Process chunks one by one with better error handling
            chunks = list(chunk_file(path))
            logger.info(f"Generated {len(chunks)} chunks from {path}")
            
            # Add batch insertion
            batch_size = 50
            batch = []
            
            for chunk_text, start_line, end_line in chunks:
                try:
                    chunk_count += 1
                    total_chunks += 1
                    
                    # Generate UUID
                    uid = generate_uuid5(f"{str(path)}:{start_line}-{end_line}")
                    
                    # Create object properly for Weaviate
                    obj = {
                        "uuid": uid,
                        "text": chunk_text,
                        "file": str(path),
                        "category": meta.get("category", "unknown"),
                        "module": meta.get("module", ""),
                        "model": meta.get("model", "")
                    }
                    
                    # Add to batch without the nested "properties" key
                    batch.append(obj)
                    
                    # Insert when batch is full
                    if len(batch) >= batch_size:
                        collection = client.collections.get("CodeChunk")
                        collection.data.insert_many(batch)
                        logger.info(f"Added batch of {len(batch)} chunks")
                        batch = []
                    
                except Exception as e:
                    logger.error(f"Error adding chunk {chunk_count} from {path}: {e}")
            
            # Insert any remaining items in batch
            if batch:
                collection = client.collections.get("CodeChunk")
                collection.data.insert_many(batch)
                logger.info(f"Added final batch of {len(batch)} chunks")
            
            # Report progress per file
            logger.info(f"Added {chunk_count} chunks from {path}")
            
            # Report progress periodically
            if total_files % 10 == 0:
                logger.info(f"Progress: processed {total_files} files, {total_chunks} chunks so far")

    logger.info(f"Completed indexing {total_chunks} chunks from {total_files} files")
    
    return file_metadata

def clean_weaviate_data():
    """Delete all existing collections from Weaviate to start fresh."""
    logger.info("Completely resetting Weaviate database...")
    
    try:
        # Get existing collection names
        collections = client.collections.list_all()
        
        # Delete each collection completely
        for collection_name in collections:
            try:
                client.collections.delete(collection_name)
                logger.info(f"✅ Deleted collection: {collection_name}")
            except Exception as e:
                logger.error(f"Error deleting collection {collection_name}: {e}")
        
        logger.info("Database reset complete - ready for fresh start")
        
    except Exception as e:
        logger.error(f"Error resetting Weaviate database: {e}")

def add_code_definition(collection, definition, subsystem=None):
    """Add code definition to Weaviate with proper error handling for duplicates."""
    try:
        # Create the definition object
        def_obj = {
            "name": definition.get("name", ""),
            "type": definition.get("type", ""),
            "file": definition.get("file", ""),
            "line": definition.get("line", 0),
            "category": "code_definition"
        }
        
        # Add additional properties if present
        if "visibility" in definition:
            def_obj["visibility"] = definition["visibility"]
        
        if "state_mutability" in definition:
            def_obj["state_mutability"] = definition["state_mutability"]
            
        if "var_type" in definition:
            def_obj["var_type"] = definition["var_type"]
            
        if "constant" in definition:
            def_obj["constant"] = definition["constant"]
            
        if "inherits_from" in definition:
            def_obj["inherits_from"] = json.dumps(definition["inherits_from"])
            
        if "modifiers" in definition:
            def_obj["modifiers"] = json.dumps(definition["modifiers"])
        
        # Add subsystem if provided
        if subsystem:
            def_obj["subsystem"] = subsystem
        
        # Generate a deterministic ID based on file+name+type to avoid duplicates
        unique_id = f"{definition.get('file', '')}-{definition.get('name', '')}-{definition.get('type', '')}"
        id = generate_uuid5(unique_id)
        
        # Check if object already exists by ID before inserting
        try:
            # First check if object exists
            existing = collection.query.fetch_object_by_id(id)
            if existing:
                logger.info(f"Definition {definition.get('name', '')} from {definition.get('file', '')} already exists, skipping.")
                return True
        except Exception:
            # Object doesn't exist, proceed with insert
            pass
            
        # Try to add the object
        collection.data.insert(
            properties=def_obj,
            uuid=id
        )
        return True
        
    except Exception as e:
        # Check for 422 error with "already exists" message
        if '422' in str(e) and 'already exists' in str(e):
            logger.info(f"Definition {definition.get('name', '')} from {definition.get('file', '')} already exists, skipping.")
            return True
        else:
            # Log other errors
            logger.warning(f"Error adding definition {definition.get('name', '')} from {definition.get('file', '')}: {e}")
            return False

def upload_code_definitions(definitions, classification_map):
    """Upload extracted code definitions to Weaviate."""
    logger.info(f"Uploading {len(definitions)} code definitions to Weaviate")
    
    collection = client.collections.get("CodeDefinition")
    
    for definition in definitions:
        # Add classification data
        path = Path(definition["file"])
        classification = classify_path(path)
        definition.update(classification)
        
        # Generate a UUID based on definition attributes
        uid = generate_uuid5(f"{definition['file']}:{definition['name']}:{definition['type']}")
        
        try:
            add_code_definition(collection, definition, definition.get("subsystem"))
        except Exception as e:
            logger.warning(f"Error adding definition {definition['name']} from {definition['file']}: {e}")
    
    logger.info(f"✅ Uploaded {len(definitions)} code definitions")

def upload_graph_relationships(graph):
    """Upload graph relationships to Weaviate."""
    logger.info(f"Uploading graph relationships to Weaviate")
    
    collection = client.collections.get("GraphRelationship")
    count = 0
    
    for source, target, data in graph.edges(data=True):
        relationship = {
            "source": source,
            "target": target,
            "type": data.get("type", "depends_on"),
            "semantic": data.get("semantic", False)
        }
        
        # Generate a UUID based on edge attributes
        uid = generate_uuid5(f"{source}:{target}:{relationship['type']}")
        
        try:
            collection.data.insert(properties=relationship, uuid=uid)
            count += 1
        except Exception as e:
            logger.warning(f"Skipping duplicate relationship: {source} → {target}")
    
    logger.info(f"✅ Uploaded {count} graph relationships")

def extract_headings_and_text(doc):
    """Extract headings and their associated text from a Word document."""
    sections = []
    current_heading = "Introduction"
    current_text = []
    
    for para in doc.paragraphs:
        # Check if this is a heading
        if para.style.name and para.style.name.startswith('Heading'):
            # Save previous section if we have text
            if current_text:
                sections.append({
                    "heading": current_heading,
                    "text": "\n".join(current_text)
                })
            
            # Start new section
            current_heading = para.text
            current_text = []
        else:
            # Add to current section if paragraph has text
            if para.text.strip():
                current_text.append(para.text)
    
    # Add the last section
    if current_text:
        sections.append({
            "heading": current_heading,
            "text": "\n".join(current_text)
        })
    
    return sections

def chunk_text(text, size=DOC_CHUNK_SIZE, overlap=DOC_OVERLAP):
    """Split text into chunks of specified size with overlap."""
    chunks = []
    start = 0
    
    # Check for empty or very short text
    if not text or len(text) < size:
        return [text] if text else []
    
    # Try to split on sentence boundaries where possible
    sentences = sent_tokenize(text)
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        # Always add at least one sentence to avoid empty chunks
        if current_length + len(sentence) > size and current_length > 0:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            
            # Create overlap by keeping some sentences
            overlap_sentences = []
            overlap_length = 0
            
            # Work backwards to include sentences up to the overlap size
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_length += len(s) + 1  # +1 for space
                else:
                    break
            
            # Start new chunk with overlapping sentences
            current_chunk = overlap_sentences
            current_length = overlap_length
        
        current_chunk.append(sentence)
        current_length += len(sentence) + 1  # +1 for space
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def tag_onboarding_chunk(chunk_text, source_file, default_model):
    """Use OpenAI to tag a document chunk with metadata."""
    prompt = f"""You're tagging Mercata onboarding documentation for smart contract and blockchain flows. Classify the following chunk:

1. What model it describes (choose one): "erc20" / "utxo" / "general"
2. The use case it supports (1-3 words max): e.g. "minting", "transfer", "listing", "redemption", "dashboard", "authentication", "pauth", "indexing", "cirrus", "query"
3. The section or topic name (short title or inferred)

Special instructions:
- If the text discusses Cirrus (the indexing component that provides SQL-like queries to blockchain data), tag it with usecase "cirrus" or "indexing"
- For discussions of event logs or contract state indexing, use usecase "indexing"
- For SQL-like queries to get blockchain data, use usecase "query"

File source: {source_file}
Default model: {default_model}

Chunk:
{chunk_text}

Return JSON with keys: model, usecase, section. Keep responses very concise.
"""
    
    try:
        # Updated OpenAI API call for v1.0.0+
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You analyze document chunks and return structured metadata as JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        result = json.loads(response.choices[0].message.content)
        logger.debug(f"Tagged chunk with metadata: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error tagging chunk: {e}")
        # Return default values if tagging fails
        return {
            "model": default_model,
            "usecase": "general",
            "section": "documentation"
        }

def process_onboarding_docs():
    """Process the onboarding .docx files and embed them in Weaviate."""
    logger.info("Processing onboarding documentation files")
    total_chunks = 0
    
    collection = client.collections.get("MercataOnboardingDoc")
    
    for doc_file, default_model in DOCS:
        doc_path = Path("other") / doc_file
        
        if not doc_path.exists():
            logger.warning(f"Document {doc_path} not found, skipping.")
            continue
        
        logger.info(f"Processing document: {doc_path}")
        
        try:
            doc = Document(doc_path)
            
            # Extract sections with headings
            sections = extract_headings_and_text(doc)
            logger.info(f"Extracted {len(sections)} sections from {doc_file}")
            
            # Process each section
            for section_idx, section in enumerate(sections):
                heading = section["heading"]
                section_text = section["text"]
                
                # Create chunks from section
                chunks = chunk_text(section_text)
                logger.info(f"Created {len(chunks)} chunks from section '{heading}'")
                
                # Process each chunk
                for chunk_idx, text_chunk in enumerate(chunks):
                    # Tag chunk with metadata using OpenAI
                    metadata = tag_onboarding_chunk(text_chunk, doc_file, default_model)
                    
                    # Create object for Weaviate
                    obj = {
                        "text": text_chunk,
                        "source": str(doc_file),
                        "model": metadata["model"],
                        "section": metadata["section"],
                        "usecase": metadata["usecase"],
                        "heading": heading,
                        "chunk_id": chunk_idx
                    }
                    
                    # Generate a UUID based on document details
                    uid = generate_uuid5(f"{doc_file}:{heading}:{chunk_idx}")
                    
                    try:
                        collection.data.insert(properties=obj, uuid=uid)
                        total_chunks += 1
                        logger.debug(f"Added document chunk: {uid[:8]}... from {doc_file}")
                    except Exception as e:
                        logger.warning(f"Skipping duplicate doc chunk: {uid[:8]}... from {doc_file}")
        
        except Exception as e:
            logger.error(f"Error processing document {doc_path}: {e}")
    
    logger.info(f"✅ Processed {total_chunks} total document chunks")
    return total_chunks

def create_doc_code_relationships(definitions, graph):
    """Create relationships between code definitions and documentation chunks."""
    logger.info("Creating relationships between code and documentation")
    
    # Get all documentation chunks
    try:
        doc_collection = client.collections.get("MercataOnboardingDoc")
        # Fixed Weaviate v4 API call to get all objects
        resp = doc_collection.query.fetch_objects()
        doc_chunks = resp.objects
        
        if not doc_chunks:
            logger.warning("No documentation chunks found, skipping relationship creation")
            return 0
        
        logger.info(f"Found {len(doc_chunks)} documentation chunks to analyze")
        
        # Extract important function and contract names
        code_entities = {
            d["name"]: {
                "type": d["type"],
                "category": d.get("category", "unknown"),
                "module": d.get("module", ""),
                "model": d.get("model", "")
            }
            for d in definitions
        }
        
        # Track created relationships to avoid duplicates
        created_relationships = set()
        relationship_count = 0
        
        relation_collection = client.collections.get("GraphRelationship")
        
        # For each doc chunk, check which code entities it references
        for chunk in doc_chunks:
            chunk_text = chunk.properties["text"].lower()
            # Convert UUID to string
            chunk_id = str(chunk.uuid)
            model = chunk.properties["model"]
            
            # Find code entities mentioned in this chunk
            for entity_name, entity_data in code_entities.items():
                # Skip very short names (to avoid false positives)
                if len(entity_name) < 5:
                    continue
                    
                # Check if the entity name appears in the chunk
                if entity_name.lower() in chunk_text:
                    # Make sure entity model matches doc model, if specified
                    if entity_data.get("model") and model != "general" and entity_data["model"] != model:
                        continue
                        
                    relationship_key = f"{entity_name}:{chunk_id}"
                    if relationship_key not in created_relationships:
                        # Create the relationship
                        relationship = {
                            "source": entity_name,
                            "target": chunk_id,
                            "type": "explained_by"
                        }
                        
                        # Generate UUID for this relationship
                        rel_uuid = generate_uuid5(f"rel:{entity_name}:{chunk_id}")
                        
                        try:
                            relation_collection.data.insert(properties=relationship, uuid=rel_uuid)
                            created_relationships.add(relationship_key)
                            relationship_count += 1
                            logger.debug(f"Created relationship: {entity_name} → document chunk {chunk_id[:8]}")
                        except Exception as e:
                            logger.debug(f"Skipping relationship: {entity_name} → document chunk {chunk_id[:8]}: {e}")
        
        logger.info(f"✅ Created {relationship_count} relationships between code and documentation")
        return relationship_count
        
    except Exception as e:
        logger.error(f"Error creating doc-code relationships: {e}")
        return 0

def create_prioritized_search(query_text):
    """Create a prioritized search strategy based on the query content."""
    
    # Determine query type/category
    query_lower = query_text.lower()
    is_contract_query = any(term in query_lower for term in ["contract", "sol", "solidity", "erc20", "utxo"])
    is_vm_query = any(term in query_lower for term in ["solidvm", "solid vm", "record", "vm", "typechecker"])
    
    # Create tiered search weights
    search_weights = {
        "docs_md": 1.0,      # Documentation markdown (highest priority)
        "docs_docx": 0.8,    # Documentation docx files (second priority) 
        "marketplace": 0.6,  # Marketplace files (third priority)
        "haskell": 0.4,      # Haskell files (lowest priority)
        "contracts": 0.9     # Contract files (high priority for contract queries)
    }
    
    # Adjust weights based on query type
    if is_contract_query:
        search_weights["contracts"] = 1.2  # Boost contract files above everything else
    if is_vm_query:
        search_weights["haskell"] = 1.1    # Boost Haskell files for VM queries
    
    # Create category filters for each tier
    filters = {
        "docs_md": [
            Filter.by_property("file").contains("docs-website") & 
            Filter.by_property("file").contains(".md")
        ],
        "docs_docx": [
            Filter.by_property("file").contains("other") & 
            Filter.by_property("file").contains(".docx")
        ],
        "marketplace": [
            Filter.by_property("file").contains("marketplace")
        ],
        "haskell": [
            Filter.by_property("file").contains(".hs")
        ],
        "contracts": [
            Filter.by_property("file").contains(".sol") | 
            (Filter.by_property("category").equal("contract"))
        ]
    }
    
    return search_weights, filters

def search_with_prioritization(query_text, limit=5):
    """Search the codebase with the prioritized strategy."""
    search_weights, filters = create_prioritized_search(query_text)
    
    # Check if this is a file search query
    file_search_indicators = ["file", "find file", "locate file", "show file", "get file"]
    is_file_search = any(query_text.lower().startswith(indicator) for indicator in file_search_indicators)
    
    if is_file_search:
        # Extract file name from query
        for indicator in file_search_indicators:
            if query_text.lower().startswith(indicator):
                file_name = query_text[len(indicator):].strip()
                return find_file_in_codebase(file_name, limit=limit)
    
    # Initialize collector for results
    all_results = []
    
    # Get collections
    code_chunk = client.collections.get("CodeChunk") 
    code_def = client.collections.get("CodeDefinition")
    doc_chunk = client.collections.get("MercataOnboardingDoc")
    
    # First search based on priority in specific categories
    for category, weight in sorted(search_weights.items(), key=lambda x: x[1], reverse=True):
        if weight <= 0:
            continue
            
        # Skip categories that don't apply to this query
        if category == "contracts" and "contract" not in query_text.lower():
            continue
        
        # Search in code chunks
        if category in filters:
            for filter_query in filters[category]:
                try:
                    results = code_chunk.query.near_text(
                        query=query_text,
                        limit=limit,
                        filters=filter_query
                    )
                    
                    if results and results.objects:
                        # Add category and weight to results for ranking
                        for obj in results.objects:
                            obj_dict = {
                                "uuid": obj.uuid,
                                "properties": obj.properties,
                                "search_category": category,
                                "search_weight": weight,
                                "_additional": {"certainty": obj.metadata.certainty}
                            }
                            all_results.append(obj_dict)
                except Exception as e:
                    logger.error(f"Error searching code chunks with filter {category}: {e}")
        
        # Also search in code definitions if applicable
        if category in ["contracts", "haskell"]:
            for filter_query in filters[category]:
                try:
                    results = code_def.query.near_text(
                        query=query_text,
                        limit=limit,
                        filters=filter_query
                    )
                    
                    if results and results.objects:
                        for obj in results.objects:
                            obj_dict = {
                                "uuid": obj.uuid,
                                "properties": obj.properties,
                                "search_category": f"{category}_def",
                                "search_weight": weight * 1.1,  # Slightly boost definitions
                                "_additional": {"certainty": obj.metadata.certainty}
                            }
                            all_results.append(obj_dict)
                except Exception as e:
                    logger.error(f"Error searching code definitions with filter {category}: {e}")
    
    # Also search documentation chunks regardless of category
    try:
        doc_results = doc_chunk.query.near_text(
            query=query_text,
            limit=limit
        )
        
        if doc_results and doc_results.objects:
            for obj in doc_results.objects:
                # Weight docs based on type
                source = obj.properties.get("source", "")
                obj_dict = {
                    "uuid": obj.uuid,
                    "properties": obj.properties,
                    "_additional": {"certainty": obj.metadata.certainty}
                }
                
                if "docx" in source:
                    obj_dict["search_category"] = "docs_docx"
                    obj_dict["search_weight"] = search_weights["docs_docx"]
                else:
                    obj_dict["search_category"] = "docs_md"
                    obj_dict["search_weight"] = search_weights["docs_md"]
                all_results.append(obj_dict)
    except Exception as e:
        logger.error(f"Error searching documentation: {e}")
    
    # Special case: Infer file search for high-level concepts (e.g., "escrow")
    # Check if query might be looking for a specific contract or component
    if any(term in query_text.lower() for term in ["escrow", "auction", "token", "nft", "marketplace", "contract"]):
        try:
            # Try to find relevant file names directly
            possible_files = []
            for term in ["escrow", "auction", "token", "nft", "marketplace"]:
                if term in query_text.lower():
                    # Look for files that might match what user is asking about
                    file_matches = find_file_in_codebase(f"{term.capitalize()}.sol", limit=3)
                    possible_files.extend(file_matches)
            
            # Add these as potential results
            for obj in possible_files:
                obj_dict = {
                    "uuid": obj.uuid,
                    "properties": obj.properties,
                    "search_category": "inferred_file",
                    "search_weight": 1.2,  # High weight for inferred files
                    "_additional": {"certainty": 0.9}  # Arbitrary high certainty
                }
                all_results.append(obj_dict)
        except Exception as e:
            logger.error(f"Error in file inference search: {e}")
    
    # Rank results by search weight * similarity score
    for result in all_results:
        result["final_score"] = result["search_weight"] * result.get("_additional", {}).get("certainty", 0.5)
    
    # Sort by final score and return top results
    sorted_results = sorted(all_results, key=lambda x: x.get("final_score", 0), reverse=True)
    return sorted_results[:limit]

def add_relationship(collection, source, target, rel_type, description=None):
    """Add relationship between code components with proper error handling for duplicates."""
    try:
        # Create relationship object
        rel_obj = {
            "source": source,
            "target": target,
            "type": rel_type
        }
        
        if description:
            rel_obj["description"] = description
        
        # Generate a deterministic ID based on source+target+type
        unique_id = f"{source}-{target}-{rel_type}"
        id = generate_uuid5(unique_id)
        
        # Check if relationship already exists by ID before inserting
        try:
            # First check if object exists
            existing = collection.query.fetch_object_by_id(id)
            if existing:
                return True  # Already exists, silently continue
        except Exception:
            # Object doesn't exist, proceed with insert
            pass
            
        # Try to add the object
        collection.data.insert(
            properties=rel_obj,
            uuid=id
        )
        return True
        
    except Exception as e:
        # Check for 422 error with "already exists" message
        if '422' in str(e) and 'already exists' in str(e):
            return True  # Already exists, silently continue
        else:
            logger.warning(f"Error adding relationship {source} → {target} ({rel_type}): {e}")
            return False

def diagnostic_find_escrow_file():
    """
    Diagnostic function to explicitly find Escrow.sol files in the database.
    This can be called after embedding to verify how the file was indexed.
    """
    logger.info("=== RUNNING DIAGNOSTIC SEARCH FOR ESCROW.SOL ===")
    collection = client.collections.get("CodeChunk")
    
    # APPROACH 1: Try different case variations
    variations = ["Escrow.sol", "escrow.sol", "ESCROW.SOL"]
    for variation in variations:
        # Try exact match with file filter
        file_filter = weaviate.classes.query.Filter.by_property("file").like(f"*{variation}*")
        results = collection.query.fetch_objects(
            filters=file_filter,
            limit=10,
            return_properties=["text", "file", "category", "module"]
        ).objects
        
        logger.info(f"Search for '*{variation}*' in file property found: {len(results)} results")
        if results:
            for r in results[:3]:  # Show first 3 matches
                logger.info(f"  MATCH: {r.properties.get('file', 'unknown')}")
    
    # APPROACH 2: Try text content search
    text_filter = weaviate.classes.query.Filter.by_property("text").like("*escrow*")
    results = collection.query.fetch_objects(
        filters=text_filter,
        limit=10,
        return_properties=["text", "file", "category", "module"]
    ).objects
    
    logger.info(f"Search for 'escrow' in text content found: {len(results)} results")
    if results:
        for r in results[:3]:  # Show first 3 matches
            logger.info(f"  MATCH: {r.properties.get('file', 'unknown')}")
    
    # APPROACH 3: Get ALL files and search manually
    results = collection.query.fetch_objects(
        limit=500,  # Get a large sample
        return_properties=["file"]
    ).objects
    
    escrow_matches = []
    for r in results:
        if "file" in r.properties:
            file_path = r.properties["file"]
            if "escrow" in file_path.lower():
                escrow_matches.append(file_path)
    
    logger.info(f"Manual search through all files found {len(escrow_matches)} with 'escrow' in path")
    for path in escrow_matches[:10]:  # Show first 10
        logger.info(f"  ESCROW FILE: {path}")
    
    logger.info("=== DIAGNOSTIC SEARCH COMPLETE ===")
    return escrow_matches

# Add this to the main function to run the diagnostic after embedding
def main():
    # ... existing code ...
    
    # After scan_and_upload() and before search_with_prioritization()
    if USE_WEAVIATE:
        init_schema()
        file_metadata = scan_and_upload()
        
        # Add diagnostic search for Escrow.sol after embedding
        escrow_files = diagnostic_find_escrow_file()
        
        # Log paths that contain Escrow files for manual verification
        if escrow_files:
            logger.info("=== USE THESE PATHS IN QUERY_CODEBASE_WEAVIATE.PY ===")
            for path in escrow_files:
                logger.info(f"Add to query detection: '{path}'")

def print_weaviate_data_summary():
    """Print a summary of all data stored in Weaviate collections."""
    logger.info("=== WEAVIATE DATA SUMMARY ===")
    
    # Get all collections
    collections = client.collections.list_all()
    logger.info(f"Found {len(collections)} collections: {collections}")
    
    # For each collection, print summary and sample data
    for collection_name in collections:
        try:
            collection = client.collections.get(collection_name)
            
            # Get total count of objects
            count_result = collection.aggregate.over_all(total_count=True)
            total_count = count_result.total_count
            
            logger.info(f"\nCollection '{collection_name}' has {total_count} objects")
            
            # Get sample objects (limit to a reasonable number)
            sample_size = min(10, total_count)
            if sample_size > 0:
                sample_objects = collection.query.fetch_objects(
                    limit=sample_size,
                    return_properties=["*"]  # Get all properties
                ).objects
                
                logger.info(f"Sample data from '{collection_name}' ({sample_size} items):")
                for i, obj in enumerate(sample_objects):
                    logger.info(f"  Item {i+1}/{sample_size}:")
                    for prop, value in obj.properties.items():
                        # Truncate very long values
                        if isinstance(value, str) and len(value) > 300:
                            logger.info(f"    {prop}: {value[:300]}... [truncated]")
                        else:
                            logger.info(f"    {prop}: {value}")
            
            # For GraphRelationship collection, show relationship types distribution
            if collection_name == "GraphRelationship" and total_count > 0:
                try:
                    rel_types = collection.aggregate.over_all(
                        group_by="type"
                    )
                    logger.info("Relationship types distribution:")
                    for group in rel_types.groups:
                        logger.info(f"  {group.value}: {group.total_count} relationships")
                except Exception as e:
                    logger.error(f"Error getting relationship types: {e}")
            
        except Exception as e:
            logger.error(f"Error getting data for collection '{collection_name}': {e}")
    
    logger.info("=== END OF WEAVIATE DATA SUMMARY ===")

if __name__ == "__main__":
    logger.info("Starting codebase and documentation embedding process")
    
    try:
        # Add timeouts to potentially slow operations
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out")
        
        # Clean up existing data first
        logger.info("Step 1/7: Cleaning existing data...")
        clean_weaviate_data()
        
        # Initialize schema
        logger.info("Step 2/7: Initializing schema...")
        init_schema()
        
        # Set up Tree-sitter parsers
        logger.info("Step 3/7: Setting up code parsers...")
        parsers = setup_tree_sitter()
        
        # Extract code graph
        logger.info("Step 4/7: Extracting code graph...")
        definitions, graph = extract_code_graph(parsers)
        
        # Scan and upload code chunks
        logger.info("Step 5/7: Scanning and uploading code chunks...")
        file_metadata = scan_and_upload()
        
        # Add debug printout to verify Escrow.sol is being indexed
        print("\n>>> FILES INDEXED (first 20):")
        indexed_files = list(file_metadata.keys())
        print("\n".join(indexed_files[:20]))
        # Check specifically for Escrow.sol
        escrow_files = [f for f in indexed_files if "escrow" in f.lower()]
        if escrow_files:
            print("\n>>> FOUND ESCROW FILES:")
            print("\n".join(escrow_files))
        else:
            print("\n>>> WARNING: No Escrow.sol files found in indexed files!")
        
        # Upload code definitions and relationships
        logger.info("Step 6/7: Uploading code definitions and relationships...")
        upload_code_definitions(definitions, file_metadata)
        upload_graph_relationships(graph)
        
        # Process and upload documentation
        logger.info("Step 7/7: Processing documentation...")
        
        # Set a reasonable timeout for OpenAI calls
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minute timeout for doc processing
        
        try:
            doc_chunks = process_onboarding_docs()
            # Turn off alarm
            signal.alarm(0)
        except TimeoutError:
            logger.error("Documentation processing timed out, skipping remaining docs")
            doc_chunks = 0
        
        logger.info("Creating relationships between code and documentation...")
        doc_code_rels = create_doc_code_relationships(definitions, graph)
        
        logger.info("✅ Codebase and documentation successfully indexed and linked")
        
        # Print a summary
        print("\n=== ANALYSIS SUMMARY ===")
        print(f"Indexed {len(file_metadata)} files")
        print(f"Extracted {len(definitions)} code definitions")
        print(f"Identified {len(graph.edges())} relationships between components")
        print(f"Processed {doc_chunks} documentation chunks")
        print(f"Created {doc_code_rels} relationships between code and documentation")
        
        # Summary by definition type
        def_types = {}
        for d in definitions:
            def_types[d["type"]] = def_types.get(d["type"], 0) + 1
        
        print("\nDefinition types:")
        for type_name, count in def_types.items():
            print(f"  {type_name}: {count}")
        
        # Summary by category
        categories = {}
        for file_path in file_metadata:
            category = file_metadata[file_path]["category"]
            categories[category] = categories.get(category, 0) + 1
        
        print("\nFiles by category:")
        for category, count in categories.items():
            print(f"  {category}: {count} files")
        print("===========================\n")
        
        # Print detailed Weaviate data summary at the end
        print_weaviate_data_summary()
        
    except Exception as e:
        logger.error(f"Failed to complete indexing process: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Close the Weaviate connection
        client.close()
        logger.info("Weaviate connection closed properly")
        