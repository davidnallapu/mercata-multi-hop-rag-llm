import os
import logging
import json
from dotenv import load_dotenv
import weaviate
# Remove Google AI imports if not needed
# from google import genai
# from google.genai import types
import requests
import re
from openai import OpenAI  # Updated import

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Weaviate client
try:
    client = weaviate.connect_to_custom(
        http_host="localhost",
        http_port=8080,
        http_secure=False,
        grpc_host="localhost",
        grpc_port=50051,
        grpc_secure=False,
        skip_init_checks=True
    )
    logger.info("Connected to Weaviate successfully")
except Exception as e:
    logger.error(f"Failed to connect to Weaviate: {e}")
    logger.info("Try running: pip install weaviate-client")
    exit(1)

def query_openai_api_direct(prompt, model="gpt-4o"):
    """Query OpenAI API directly using HTTP requests instead of the SDK."""
    logger.info(f"Querying OpenAI API directly with model: {model}")
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY not found in environment variables."
        
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant that provides accurate information about code."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 4000,
            "temperature": 0.3
        }
        
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Raise exception for non-200 status codes
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except Exception as e:
        logger.error(f"Failed to call OpenAI API directly: {e}")
        return f"API call failed: {str(e)}"

def query_openai_api(prompt, model="gpt-4o"):
    """Query OpenAI API using their SDK."""
    logger.info(f"Querying OpenAI API with model: {model}")
    
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Error: OPENAI_API_KEY not found in environment variables."
        
        # Create client with API key
        client = OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides accurate information about code."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Failed to call OpenAI API: {e}")
        return f"API call failed: {str(e)}"

def query_codebase_structure(query, limit=10):
    """
    Query both CodeDefinitions and GraphRelationships explicitly.
    Returns entities and their relationships from the knowledge graph.
    """
    try:
        # Get code definitions
        code_def_coll = client.collections.get("CodeDefinition")
        entities = code_def_coll.query.near_text(
            query=query,
            limit=limit,
            return_properties=["name", "type", "file", "importance", "subsystem"]
        ).objects
        
        # Get relationships
        relationships = []
        graph_rel_coll = client.collections.get("GraphRelationship")
        
        # For each entity, find its relationships
        for entity in entities:
            if 'name' not in entity.properties:
                continue
                
            name = entity.properties["name"]
            
            # Find relationships where this entity is the source - using Weaviate v4 filter syntax
            source_filter = weaviate.classes.query.Filter.by_property("source").equal(name)
            rels_from = graph_rel_coll.query.fetch_objects(
                filters=source_filter,
                limit=limit,
                return_properties=["source", "target", "type"]
            ).objects
            
            # Find relationships where this entity is the target - using Weaviate v4 filter syntax
            target_filter = weaviate.classes.query.Filter.by_property("target").equal(name)
            rels_to = graph_rel_coll.query.fetch_objects(
                filters=target_filter,
                limit=limit,
                return_properties=["source", "target", "type"]
            ).objects
            
            relationships.extend(rels_from + rels_to)
        
        return entities, relationships
    except Exception as e:
        logger.warning(f"Error querying codebase structure: {e}")
        return [], []

def query_with_context(user_query, num_chunks=5, max_tokens=8192, use_direct_api=False, use_openai=True, min_iterations=5, max_iterations=10, verbose=True, openai_model="gpt-4o"):
    """Query language models with context from the Weaviate database with iterative refinement."""
    logger.info(f"Querying with: '{user_query}'")
    
    try:
        # Initialize chunks as an empty list
        chunks = []
        
        # AUTO-DETECT FILE SEARCH QUERIES - Add this block
        file_extensions = ['.sol', '.hs', '.js', '.jsx', '.ts', '.tsx', '.json', '.md']
        file_search_indicators = ["file", "find file", "locate file", "show file", "get file"]
        
        # Check if query is already a file search query
        is_already_file_search = any(user_query.lower().startswith(indicator) for indicator in file_search_indicators)
        
        # If not already a file search query, check if it looks like one
        if not is_already_file_search:
            # Extract filename pattern from the query
            filename_patterns = [
                r'(?:^|\s)([a-zA-Z0-9_-]+\.sol)',  # .sol files
                r'(?:^|\s)([a-zA-Z0-9_-]+\.hs)',   # .hs files
                r'(?:^|\s)([a-zA-Z0-9_-]+\.[jt]sx?)',  # .js, .jsx, .ts, .tsx files
                r'(?:^|\s)([a-zA-Z0-9_-]+\.md)',   # .md files
                r'(?:^|\s)([a-zA-Z0-9_-]+\.json)',  # .json files
                r'(?:^|\s)(Escrow\.sol|Token\.sol|Auction\.sol)'  # Specific known files
            ]
            
            filename = None
            for pattern in filename_patterns:
                match = re.search(pattern, user_query, re.IGNORECASE)
                if match:
                    filename = match.group(1)
                    break
            
            # If we found a filename in the query
            if filename:
                original_query = user_query
                user_query = f"find file {filename}"
                if verbose:
                    print(f"\n[Auto-detected file search. Extracting '{filename}' from query. Converting to '{user_query}']")
                logger.info(f"Auto-converted to file search query: '{user_query}'")
        
        # NEW: SPECIAL HANDLING FOR FILE SEARCHES
        # Check if this is a file search query after auto-detection
        if any(user_query.lower().startswith(indicator) for indicator in file_search_indicators):
            # Extract the filename from the query
            filename = user_query.split(' ', 2)[-1].strip()
            if verbose:
                print(f"\n[Performing direct file search for: {filename}]")
            logger.info(f"Direct file search for: {filename}")
            
            # Get collection
            collection = client.collections.get("CodeChunk")
            
            # Try multiple search strategies for finding the file
            chunks = []
            
            # Strategy 1: Direct case-insensitive search in file property
            file_filter = weaviate.classes.query.Filter.by_property("file").like(f"*{filename}*")
            results = collection.query.fetch_objects(
                filters=file_filter,
                limit=num_chunks * 2,
                return_properties=["text", "file", "category", "module"]
            ).objects
            
            if results:
                chunks.extend(results)
                if verbose:
                    print(f"\n[Found {len(results)} chunks with filename pattern: {filename}]")
                logger.info(f"Found {len(results)} chunks with filename pattern: {filename}")
            
            # Strategy 2: Check if filename is Escrow.sol - special handling for known paths
            if filename.lower() == "escrow.sol":
                # Try common paths where Escrow.sol might be
                known_paths = [
                    "*mercata*escrow*",
                    "*contracts*escrow*",
                    "*templates*escrow*",
                    "*escrow*.sol",
                    "*Escrow*.sol"
                ]
                
                for path_pattern in known_paths:
                    path_filter = weaviate.classes.query.Filter.by_property("file").like(path_pattern)
                    path_results = collection.query.fetch_objects(
                        filters=path_filter,
                        limit=5,
                        return_properties=["text", "file", "category", "module"]
                    ).objects
                    
                    if path_results:
                        chunks.extend(path_results)
                        if verbose:
                            print(f"\n[Found {len(path_results)} chunks with path pattern: {path_pattern}]")
                        logger.info(f"Found {len(path_results)} chunks with path pattern: {path_pattern}")
            
            # Strategy 3: Content-based search as fallback
            if not chunks and filename.lower() == "escrow.sol":
                content_filter = weaviate.classes.query.Filter.by_property("text").like("*escrow*")
                content_results = collection.query.fetch_objects(
                    filters=content_filter,
                    limit=num_chunks,
                    return_properties=["text", "file", "category", "module"]
                ).objects
                
                if content_results:
                    chunks.extend(content_results)
                    if verbose:
                        print(f"\n[Found {len(content_results)} chunks with 'escrow' in content]")
                    logger.info(f"Found {len(content_results)} chunks with 'escrow' in content")
            
            # If we found chunks, skip the regular vector search
            if chunks:
                if verbose:
                    print(f"\n[Using direct file search results instead of vector search]")
                logger.info(f"Using direct file search results: {len(chunks)} chunks")
            else:
                # Fall back to regular vector search if all strategies failed
                if verbose:
                    print(f"\n[Direct file search found no results, falling back to vector search]")
                logger.info(f"File search found no results, falling back to vector search")
        
        # Only do regular vector search if we don't already have chunks from file search
        if not chunks:
            # Get initial relevant code chunks from Weaviate - fixed property list
            collection = client.collections.get("CodeChunk")
            vector_results = collection.query.near_text(
                query=user_query,
                limit=num_chunks * 2,  # Increase initial chunk count for better context
                return_properties=["text", "file", "category", "module"]
            )
            
            chunks = vector_results.objects
        
        if not chunks:
            return "No relevant code found to answer your question."
        
        # NEW: Search for language definitions if keywords are mentioned
        keyword_match = re.search(r'keyword[s]?\s+["\']?(\w+)["\']?', user_query, re.IGNORECASE)
        if keyword_match:
            keyword = keyword_match.group(1)
            logger.info(f"Detected keyword search for: {keyword}")
            
            # First, search for parser files - updated for v4 syntax
            parser_filters = collection.query.filter.by_property("file").like(f"*Parse*") & collection.query.filter.by_property("text").like(f"*{keyword}*")
            
            parser_results = collection.query.fetch_objects(
                filters=parser_filters,
                limit=3,
                return_properties=["text", "file", "category", "module"]
            )
            
            parser_chunks = parser_results.objects
            
            # Also look in declaration files - updated for v4 syntax
            decl_filters = collection.query.filter.by_property("file").like(f"*Declaration*") & collection.query.filter.by_property("text").like(f"*{keyword}*")
            
            decl_results = collection.query.fetch_objects(
                filters=decl_filters,
                limit=3,
                return_properties=["text", "file", "category", "module"]
            )
            
            decl_chunks = decl_results.objects
            
            # Combine parser and declaration chunks
            language_def_chunks = parser_chunks + decl_chunks
            
            if language_def_chunks:
                logger.info(f"Found {len(language_def_chunks)} language definition chunks for keyword '{keyword}'")
                chunks.extend(language_def_chunks)
        
        # Track all retrieved chunks to avoid duplicates
        all_chunk_ids = set([chunk.properties.get('id', hash(chunk.properties.get('text', ''))) for chunk in chunks])
        retrieved_files = set([chunk.properties.get('file', '') for chunk in chunks])
        
        # Get structural information from the knowledge graph
        try:
            entities, relationships = query_codebase_structure(user_query)
            has_structure_info = len(entities) > 0 or len(relationships) > 0
            logger.info(f"Found {len(entities)} entities and {len(relationships)} relationships")
        except Exception as e:
            logger.warning(f"Could not fetch knowledge graph structure: {e}")
            entities = []
            relationships = []
            has_structure_info = False
        
        # Initial context building - ensure full chunks are provided
        context = "Here are code chunks that might be relevant to your question:\n\n"
        for i, chunk in enumerate(chunks, 1):
            file_path = chunk.properties.get('file', 'unknown')
            category = chunk.properties.get('category', 'unknown')
            # Display full chunk instead of truncating at 1000 chars
            context += f"CHUNK {i} (from {file_path}, category: {category}):\n```\n{chunk.properties.get('text', '')}\n```\n\n"
        
        # Add structural information if available
        if has_structure_info:
            context += "\nCODEBASE STRUCTURE INFORMATION:\n"
            
            if entities:
                context += "\nRELEVANT CODE ENTITIES:\n"
                for entity in entities[:7]:  # Limit to not overwhelm
                    context += f"- {entity.properties.get('name', 'Unknown')} ({entity.properties.get('type', 'Unknown')})"
                    if 'file' in entity.properties:
                        context += f" in {entity.properties.get('file')}"
                    if 'importance' in entity.properties:
                        context += f" [importance: {entity.properties.get('importance')}]"
                    context += "\n"
            
            if relationships:
                context += "\nRELATIONSHIPS BETWEEN COMPONENTS:\n"
                for rel in relationships[:10]:  # Limit to top 10
                    rel_type = rel.properties.get('type', 'related to')
                    source = rel.properties.get('source', 'Unknown')
                    target = rel.properties.get('target', 'Unknown')
                    desc = rel.properties.get('description', '')
                    context += f"- {source} → {target}: {rel_type}"
                    if desc:
                        context += f" ({desc})"
                    context += "\n"
        
        # Create the initial prompt with enhanced instructions for the LLM
        system_instructions = (
            "You are an expert Developer Relations (DevRel) professional for STRATO blockchain technology, speaking with "
            "confidence and authority. You're enthusiastic about STRATO's features and capabilities, and you excel at explaining "
            "technical concepts in a clear, practical way. Provide definitive answers using the available code context, "
            "highlighting best practices and practical use cases whenever possible.\n\n"
            
            "EMBEDDING SYSTEM INFORMATION:\n"
            "The codebase you're querying has been indexed using a sophisticated system that:\n"
            "- Uses Tree-sitter to parse Haskell (.hs), Solidity (.sol), and JavaScript (.js/.jsx) files\n"
            "- Extracts code definitions (functions, types, contracts, etc.) and their relationships\n"
            "- Creates a knowledge graph of components and their connections\n"
            "- Processes documentation (.docx files) and links it to relevant code\n"
            "- Stores everything in a Weaviate vector database with several collections:\n"
            "  * CodeChunk: Contains code snippets with semantic vectors\n"
            "  * CodeDefinition: Contains extracted entities like functions and types\n"
            "  * GraphRelationship: Contains relationships between code components\n"
            "  * MercataOnboardingDoc: Contains documentation chunks\n\n"
            
            "SEARCH SYSTEM INFORMATION:\n"
            "You are connected to a Weaviate vector database containing code chunks from the STRATO codebase. "
            "The vector search finds semantically similar content based on your queries. To get the best results:\n"
            "- Use precise technical terms that would appear in the code or documentation\n"
            "- Try alternative technical phrasings if your first search doesn't find what you need\n"
            "- Search for specific file names when you know what you're looking for\n"
            "- Search for specific functions, methods, or contract names\n"
            "- Be aware that the database contains chunks of code, so context might be spread across multiple chunks\n\n"
            
            "LANGUAGE DEFINITION AWARENESS:\n"
            "- Pay special attention to parser files (.hs) as they define the core SolidVM language features\n"
            "- Declaration.hs and similar files contain definitive information about language keywords\n"
            "- Parser implementations are the source of truth for language features\n"
            "- Give higher weight to implementation files over documentation when they conflict\n\n"
            
            "INFORMATION SYNTHESIS INSTRUCTIONS:\n"
            "- The code chunks you receive are fragments from a larger codebase - you must CONNECT information across them\n"
            "- Look for relationships between files and how components interact with each other\n"
            "- When you find multiple fragments of the same file, mentally reconstruct the complete implementation\n"
            "- Cross-reference JavaScript files with Solidity contracts to understand the full implementation\n"
            "- Pay attention to import statements and function calls to understand dependencies\n"
            "- Always prioritize information directly related to the user's original question\n\n"
            
            "When explaining code, be specific about how things work and provide example implementations when relevant. "
            "Use a confident, knowledgeable tone that reflects your deep expertise with the STRATO ecosystem. "
            "Your goal is to help developers successfully implement their solutions with clarity and best practices.\n\n"
            
            "IMPORTANT SEARCH GUIDANCE:\n"
            "1. If you need more information, make ONE SPECIFIC request at a time using: 'NEED_MORE_INFO: [specific query]'\n"
            "2. Focus on finding SPECIFIC FILES first (e.g., 'Find Declarations.hs file')\n"
            "3. When requesting language definitions, look for parser files (.hs) and declaration files\n" 
            "4. USE ALL AVAILABLE ITERATIONS to gather comprehensive information before answering\n"
            "5. If you mention a file as important but haven't found it yet, your next query MUST search for it\n"
            "6. Do not provide a final answer until you've found ALL critical implementation files you've identified\n"
            "7. MAINTAIN FOCUS on the original question - don't fixate on tangentially related files\n"
            "8. ALWAYS EVALUATE the relevance of files you find to the original question\n\n"
            
            "UNDERSTANDING CODE STRUCTURE:\n"
            "1. The codebase is organized into subsystems with clear relationships between components\n"
            "2. Functions/methods often have relationships that can be discovered through the GraphRelationship collection\n"
            "3. When exploring a component, try to find its related components through explicit searches\n"
            "4. Pay attention to 'importance' ratings on code definitions - higher importance indicates core functionality\n"
            "5. Use both semantic searches (for concepts) and explicit file searches (for implementations)\n\n"
            
            "Good search examples:\n"
            "- 'NEED_MORE_INFO: Find the Declarations.hs parser file'\n"
            "- 'NEED_MORE_INFO: Search for keyword definitions in Parse directory'\n"
            "- 'NEED_MORE_INFO: Look for record keyword implementation in SolidVM'\n"
            "- 'NEED_MORE_INFO: Find language syntax definitions for SolidVM'\n"
            "- 'NEED_MORE_INFO: Find related components to function X using GraphRelationship collection'\n\n"
            
            "RESPONSE FORMAT REQUIREMENTS:\n"
            "1. ALWAYS include relevant code chunks in your response\n"
            "2. When referring to code, cite the specific CHUNK number\n"
            "3. Provide detailed explanations with specific function names, parameters, and return values\n"
            "4. Explain how the code works with concrete examples\n"
            "5. Your response should be detailed and comprehensive - at least 500 words\n"
            "6. Break down complex relationships between components\n"
            "7. Always show implementation details from the code chunks\n\n"
        )
        
        initial_prompt = f"{system_instructions}\n\n{context}\n\nBased on the code above, please answer this question with DevRel confidence and expertise: {user_query}\n\n" + \
                         f"Retrieved files so far: {', '.join(retrieved_files)}"
        
        conversation_history = []
        iterations = 0
        final_answer = None
        
        # Begin iterative process
        while iterations < max_iterations:
            iterations += 1
            logger.info(f"Starting iteration {iterations}/{max_iterations}")
            
            if verbose:
                print(f"\n[Thinking... iteration {iterations}/{max_iterations}]")
            
            # Get LLM response
            if use_openai:
                current_prompt = initial_prompt if iterations == 1 else f"{initial_prompt}\n\nConversation so far:\n{''.join(conversation_history)}\n\nContinue helping the user with DevRel expertise and confidence."
                
                # For early iterations, explicitly instruct the LLM to keep searching
                if iterations < min_iterations:
                    current_prompt += f"\n\nIMPORTANT: This is iteration {iterations} of a minimum of {min_iterations} required iterations. DO NOT provide a final answer yet. Instead, use 'NEED_MORE_INFO:' to request additional specific information about the codebase to build a more complete understanding."
                
                response = query_openai_api(current_prompt, model=openai_model)
            elif use_direct_api:
                current_prompt = initial_prompt if iterations == 1 else f"{initial_prompt}\n\nConversation so far:\n{''.join(conversation_history)}\n\nContinue helping the user with DevRel expertise and confidence."
                response = query_openai_api_direct(current_prompt, model=openai_model)
            else:
                # Fallback to using the OpenAI SDK
                try:
                    current_prompt = initial_prompt if iterations == 1 else f"{initial_prompt}\n\nConversation so far:\n{''.join(conversation_history)}\n\nContinue helping the user with DevRel expertise and confidence."
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        return "Error: OPENAI_API_KEY not found in environment variables."
                    
                    # Create client with API key - use a different name to avoid conflict
                    openai_client = OpenAI(api_key=api_key)
                    
                    response = openai_client.chat.completions.create(
                        model=openai_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that provides accurate information about code."},
                            {"role": "user", "content": current_prompt}
                        ],
                        max_tokens=4000,
                        temperature=0.3
                    ).choices[0].message.content
                except Exception as e:
                    logger.error(f"Failed to call OpenAI API: {e}")
                    return f"LLM failed: {str(e)}"
            
            # Check if LLM needs more information
            if "NEED_MORE_INFO:" in response:
                follow_up_query = response.split("NEED_MORE_INFO:")[1].strip()
                follow_up_query = follow_up_query.split("\n")[0].strip()  # Get just the first line
                
                if verbose:
                    print(f"\n[AI is looking for more information: {follow_up_query}]")
                
                logger.info(f"LLM requesting more info: {follow_up_query}")
                
                # NEW CODE: Apply the auto-detection to follow-up queries too
                # Check if query mentions 'file' and a filename
                file_search_indicators = ["file", "find file", "locate file", "show file", "get file"]
                is_already_file_search = any(follow_up_query.lower().startswith(indicator) for indicator in file_search_indicators)
                
                if not is_already_file_search:
                    # First check known problematic files with their exact paths
                    known_files = {
                        "escrow": "Escrow.sol",  # Will be replaced with actual paths from diagnostic
                        "escrow.sol": "Escrow.sol",
                    }
                    
                    # Update with exact paths found by diagnostic - replace these with what your diagnostic outputs
                    exact_escrow_paths = [
                        # These will be filled in based on your diagnostic output
                        "path/to/Escrow.sol",  # Replace with actual paths from diagnostic
                    ]
                    
                    # Check for escrow-related terms
                    if any(term in follow_up_query.lower() for term in ["escrow", "escrow.sol", "ecrow"]):
                        # First try to use a direct file search if we have exact paths
                        if exact_escrow_paths:
                            # Use the first path we found (most relevant)
                            original_query = follow_up_query
                            path_to_use = exact_escrow_paths[0]
                            follow_up_query = f"find file {path_to_use}"
                            if verbose:
                                print(f"[Auto-converting to direct file search with exact path: '{follow_up_query}']")
                            logger.info(f"Auto-converted follow-up to file search with exact path: '{follow_up_query}'")
                
                # Log the interaction
                conversation_history.append(f"\nLLM: {response}\n")
                
                # Get additional chunks - first try with the specific query
                new_vector_results = collection.query.near_text(
                    query=follow_up_query,
                    limit=num_chunks,
                    return_properties=["text", "file", "category", "module"]
                )
                
                new_chunks = new_vector_results.objects
                
                # If the query mentions specific directories, try to search within those directories
                dir_search = None
                if follow_up_query and ("directory" in follow_up_query.lower() or "folder" in follow_up_query.lower() or "dir" in follow_up_query.lower()):
                    try:
                        # Extract potential directory names (look for words after "directory", "folder", or "dir")
                        dir_matches = re.findall(r'(?:directory|folder|dir)\s+(\w+)', follow_up_query.lower())
                        if dir_matches:
                            dir_search = dir_matches[0]
                            logger.info(f"Detected directory search request for: {dir_search}")
                            
                            # Try to find chunks specifically from this directory - updated for Weaviate v4
                            dir_filter = weaviate.classes.query.Filter.by_property("file").like(f"*{dir_search}*")
                            dir_query = collection.query.fetch_objects(
                                filters=dir_filter,
                                limit=num_chunks,
                                return_properties=["text", "file", "category", "module"]
                            )
                            
                            dir_chunks = dir_query.objects
                            if dir_chunks:
                                new_chunks.extend(dir_chunks)
                                logger.info(f"Found {len(dir_chunks)} chunks in directory: {dir_search}")
                    except Exception as e:
                        logger.error(f"Error in directory search: {e}")
                        # Continue execution even if directory search fails
                
                # Filter out duplicates
                unique_new_chunks = []
                for chunk in new_chunks:
                    chunk_id = chunk.properties.get('id', hash(chunk.properties.get('text', '')))
                    if chunk_id not in all_chunk_ids:
                        all_chunk_ids.add(chunk_id)
                        unique_new_chunks.append(chunk)
                        retrieved_files.add(chunk.properties.get('file', ''))
                
                # Add new chunks to context
                if unique_new_chunks:
                    additional_context = "\nADDITIONAL CHUNKS FOUND:\n\n"
                    for i, chunk in enumerate(unique_new_chunks, len(chunks)+1):
                        file_path = chunk.properties.get('file', 'unknown')
                        category = chunk.properties.get('category', 'unknown')
                        additional_context += f"CHUNK {i} (from {file_path}, category: {category}):\n```\n{chunk.properties.get('text', '')}\n```\n\n"
                    
                    # Add to conversation history
                    conversation_history.append(f"\nSYSTEM: {additional_context}\n")
                    
                    if verbose:
                        print(f"\n[Found {len(unique_new_chunks)} new relevant code chunks]")
                        if dir_search:
                            print(f"[Including chunks from directory: {dir_search}]")
                else:
                    conversation_history.append("\nSYSTEM: No additional relevant chunks found.\n")
                    if verbose:
                        print("\n[No additional relevant code chunks found]")
            else:
                # LLM has provided a final answer - but check if we've met minimum iterations
                if iterations >= min_iterations:
                    final_answer = response
                    if verbose:
                        print("\n[AI has completed its answer after sufficient iterations]")
                    # Exit the loop once we have a final answer AND have met minimum iterations
                    break
                else:
                    # Force continued exploration by converting the response into a NEED_MORE_INFO request
                    logger.info(f"LLM tried to answer early at iteration {iterations}/{min_iterations}. Forcing continued exploration.")
                    if verbose:
                        print(f"\n[AI tried to answer too early. Forcing it to explore more context (iteration {iterations}/{min_iterations})]")
                    
                    # Extract a follow-up query from the response or create a general one
                    follow_up_query = f"Find more information about {user_query}"
                    
                    # Add forced exploration message to conversation history
                    conversation_history.append(f"\nLLM: {response}\n")
                    conversation_history.append(f"\nSYSTEM: You must continue exploring for at least {min_iterations} iterations. Currently at iteration {iterations}. Please find more relevant information.\n")
                    
                    # Get additional chunks with this follow-up query
                    new_vector_results = collection.query.near_text(
                        query=follow_up_query,
                        limit=num_chunks,
                        return_properties=["text", "file", "category", "module"]
                    )
                    
                    new_chunks = new_vector_results.objects
                    
                    # Filter out duplicates
                    unique_new_chunks = []
                    for chunk in new_chunks:
                        chunk_id = chunk.properties.get('id', hash(chunk.properties.get('text', '')))
                        if chunk_id not in all_chunk_ids:
                            all_chunk_ids.add(chunk_id)
                            unique_new_chunks.append(chunk)
                            retrieved_files.add(chunk.properties.get('file', ''))
                    
                    # Add new chunks to context
                    if unique_new_chunks:
                        additional_context = "\nADDITIONAL CHUNKS FOUND:\n\n"
                        for i, chunk in enumerate(unique_new_chunks, len(chunks)+1):
                            file_path = chunk.properties.get('file', 'unknown')
                            category = chunk.properties.get('category', 'unknown')
                            additional_context += f"CHUNK {i} (from {file_path}, category: {category}):\n```\n{chunk.properties.get('text', '')}\n```\n\n"
                        
                        # Add to conversation history
                        conversation_history.append(f"\nSYSTEM: {additional_context}\n")
                        
                        if verbose:
                            print(f"\n[Found {len(unique_new_chunks)} new relevant code chunks]")
                    else:
                        conversation_history.append("\nSYSTEM: No additional relevant chunks found.\n")
                        if verbose:
                            print("\n[No additional relevant code chunks found]")
        
        # Return the final answer or what we have after max iterations
        return final_answer if final_answer else response
    
    except Exception as e:
        logger.error(f"Error during query: {e}")
        return f"An error occurred: {str(e)}"

def query_codebase(query_text, limit=5):
    """
    Query the codebase using semantic search
    
    Args:
        query_text: The natural language query
        limit: Maximum number of results to return
    
    Returns:
        List of results with relevant code chunks
    """
    try:
        collection = client.collections.get("CodeChunk")
        response = collection.query.near_text(
            query=query_text,
            limit=limit,
            return_properties=["text", "file"]
        )
        
        return response.objects
    except Exception as e:
        logger.error(f"Error querying codebase: {e}")
        return []

def interactive_query_mode():
    """Interactive mode to query the codebase using LLMs."""
    print("\n=== INTERACTIVE QUERY MODE ===")
    print("Type 'exit' to quit")
    print("Type 'raw' to see raw Weaviate results instead of LLM")
    print("Type 'openai' to use OpenAI (default)")
    print("Type 'direct' to use direct API calls instead of SDK")
    print("Type 'openai-model [model]' to set OpenAI model (default: gpt-4o)")
    print("Type 'depth N' to set max iteration depth (default: 10)")
    print("Type 'min N' to set min iteration depth (default: 10)")
    print("Type 'verbose on/off' to toggle detailed progress (default: on)")
    print("Type 'chunks N' to set number of chunks to retrieve (default: 5)")
    
    use_direct_api = False
    use_openai = True
    openai_model = "gpt-4o"
    max_iterations = 10
    min_iterations = 10
    verbose = True
    num_chunks = 5
    
    while True:
        query = input("\nAsk a question about the codebase: ")
        if query.lower() in ['exit', 'quit']:
            break
            
        if query.lower() == 'raw':
            # Switch to raw mode
            raw_query = input("Enter query for raw results: ")
            results = query_codebase(raw_query)
            
            print("\n=== RAW RESULTS ===")
            for i, result in enumerate(results, 1):
                print(f"\n--- Result {i} ---")
                print(f"File: {result.properties.get('file')}")
                print(f"Snippet:\n{result.properties.get('text', '')[:300]}...")
        elif query.lower() == 'direct':
            # Toggle direct API mode
            use_direct_api = not use_direct_api
            print(f"Direct API mode {'enabled' if use_direct_api else 'disabled'}")
        elif query.lower() == 'openai':
            # Enable OpenAI mode
            use_openai = True
            use_direct_api = False
            print("OpenAI mode enabled")
        elif query.lower().startswith('openai-model '):
            # Set OpenAI model
            try:
                openai_model = query.lower().split('openai-model ')[1].strip()
                print(f"OpenAI model set to {openai_model}")
                # Enable OpenAI mode if not already enabled
                if not use_openai:
                    use_openai = True
                    print("OpenAI mode enabled")
            except (ValueError, IndexError):
                print("Invalid model name. Using default gpt-4o.")
                openai_model = "gpt-4o"
        elif query.lower().startswith('depth '):
            # Set max iteration depth
            try:
                max_iterations = int(query.lower().split('depth ')[1])
                print(f"Max iteration depth set to {max_iterations}")
            except (ValueError, IndexError):
                print("Invalid depth value. Using default of 10.")
                max_iterations = 10
        elif query.lower().startswith('min '):
            # Set min iteration depth
            try:
                min_iterations = int(query.lower().split('min ')[1])
                print(f"Min iteration depth set to {min_iterations}")
            except (ValueError, IndexError):
                print("Invalid min value. Using default of 10.")
                min_iterations = 10
        elif query.lower() == 'verbose on':
            verbose = True
            print("Verbose mode enabled")
        elif query.lower() == 'verbose off':
            verbose = False
            print("Verbose mode disabled")
        elif query.lower().startswith('chunks '):
            # Set number of chunks to retrieve
            try:
                num_chunks = int(query.lower().split('chunks ')[1])
                print(f"Number of chunks set to {num_chunks}")
            except (ValueError, IndexError):
                print("Invalid chunks value. Using default of 5.")
                num_chunks = 5
        else:
            # Use LLM for enhanced answers
            print("\nQuerying with context from codebase...")
            response = query_with_context(
                query, 
                use_direct_api=use_direct_api,
                use_openai=use_openai,
                openai_model=openai_model,
                max_iterations=max_iterations,
                min_iterations=min_iterations,
                verbose=verbose
            )
            print("\n" + "="*80)
            print("LLM RESPONSE:")
            print(response)
            print("="*80)

def create_semantic_doc_code_relationships(similarity_threshold=0.7, limit=5):
    """
    Create relationships between documentation chunks and code definitions using semantic similarity.
    
    Args:
        similarity_threshold: Minimum similarity score to create a relationship
        limit: Maximum number of code definitions to link to each doc chunk
    
    Returns:
        Number of relationships created
    """
    logger.info("Creating semantic relationships between documentation and code...")
    
    try:
        # Get documentation collection
        doc_collection = client.collections.get("DocumentationChunk")
        code_collection = client.collections.get("CodeDefinition")
        
        # Get all documentation chunks
        doc_chunks = doc_collection.query.fetch_objects(
            limit=10000,  # Adjust as needed
            return_properties=["text", "file", "id"]
        ).objects
        
        relationship_count = 0
        
        # For each doc chunk, find semantically similar code definitions
        for chunk in doc_chunks:
            chunk_text = chunk.properties.get("text", "")
            chunk_id = chunk.uuid
            
            # Use the chunk text to find similar code definitions
            code_results = code_collection.query.near_text(
                query=chunk_text,
                limit=limit,
                return_properties=["name", "type", "file"]
            )
            
            # Create relationships for matches above the threshold
            for code_entity in code_results.objects:
                # In a real implementation, you'd check the distance/similarity score
                # For Weaviate, we need to get the distance from the result
                # This is a simplified version assuming all returned results are relevant
                
                code_entity_id = code_entity.uuid
                entity_name = code_entity.properties.get("name", "")
                entity_type = code_entity.properties.get("type", "")
                
                # Create a relationship between the doc chunk and code entity
                # This would require a GraphRelationship collection
                graph_rel_coll = client.collections.get("GraphRelationship")
                
                # Create relationship object
                relationship = {
                    "source": entity_name,  # Using name as the identifier
                    "target": f"doc:{chunk_id}",  # Prefixing with 'doc:' to distinguish docs
                    "type": "explained_by",
                    "description": f"Documentation explains {entity_type} {entity_name}"
                }
                
                try:
                    graph_rel_coll.data.insert(relationship)
                    relationship_count += 1
                except Exception as e:
                    logger.warning(f"Error creating relationship: {e}")
            
            # Log progress for every 100 chunks
            if relationship_count % 100 == 0:
                logger.info(f"Created {relationship_count} relationships so far...")
        
        logger.info(f"Successfully created {relationship_count} relationships between documentation and code")
        return relationship_count
    
    except Exception as e:
        logger.error(f"Error creating semantic relationships: {e}")
        return 0

if __name__ == "__main__":
    # Database inspection
    print("\n=== WEAVIATE DATABASE INSPECTION ===")
    
    # Get all collection names in the database
    collections = client.collections.list_all()
    print(f"Collections in database: {collections}")

    # For each collection, get basic stats
    for collection_name in collections:
        collection = client.collections.get(collection_name)
        count = collection.aggregate.over_all().total_count
        print(f"Collection '{collection_name}' has {count} objects")
    
    # Inspect CodeDefinition collection if it exists
    if "CodeDefinition" in collections:
        code_def_coll = client.collections.get("CodeDefinition")
        defs = code_def_coll.query.fetch_objects(limit=10, return_properties=["name", "type", "file"])
        print("\nSample CodeDefinition objects:")
        for obj in defs.objects[:10]:
            print(f"- {obj.properties.get('name', 'N/A')} ({obj.properties.get('type', 'N/A')}) in {obj.properties.get('file', 'N/A')}")
        
        # Get all unique types of code definitions
        try:
            types_result = code_def_coll.aggregate.over("type").with_fields("groupedBy", "total_count")
            print("\nTypes of code definitions:")
            for group in types_result.groups:
                print(f"- {group.group_by['value']}: {group.total_count} items")
        except Exception as e:
            print(f"Could not aggregate types: {e}")
    
    # Inspect GraphRelationship collection if it exists
    if "GraphRelationship" in collections:
        rel_coll = client.collections.get("GraphRelationship")
        rels = rel_coll.query.fetch_objects(limit=10, return_properties=["source", "target", "type"])
        print("\nSample GraphRelationship objects:")
        for rel in rels.objects[:10]:
            print(f"- {rel.properties.get('source', 'N/A')} → {rel.properties.get('target', 'N/A')} ({rel.properties.get('type', 'N/A')})")
        
        # Get all relationship types
        try:
            rel_types_result = rel_coll.aggregate.over("type").with_fields("groupedBy", "total_count")
            print("\nTypes of relationships:")
            for group in rel_types_result.groups:
                print(f"- {group.group_by['value']}: {group.total_count} items")
        except Exception as e:
            print(f"Could not aggregate relationship types: {e}")
    
    print("\n=== END OF DATABASE INSPECTION ===\n")
    
    # Check if Weaviate has data
    try:
        # Check if CodeChunk collection exists - fixed for Weaviate v4
        collections = client.collections.list_all()
        collection_names = [c for c in collections]  # In v4, these are already strings
        has_codechunk = "CodeChunk" in collection_names
        
        if not has_codechunk:
            logger.error("CodeChunk class not found in Weaviate. Please run embed_codebase_weaviate.py first to index your code.")
            exit(1)
        
        # Check if we have data
        collection = client.collections.get("CodeChunk")
        # Corrected for Weaviate v4 - using aggregate API
        count = collection.aggregate.over_all().total_count
        
        if count == 0:
            logger.error("No code chunks found in Weaviate. Please run embed_codebase_weaviate.py first to index your code.")
            exit(1)
            
        logger.info(f"Found {count} code chunks in Weaviate")
        
    except Exception as e:
        logger.error(f"Error checking Weaviate schema: {e}")
        logger.error("Make sure Weaviate is running and accessible at http://localhost:8080")
        exit(1)
    
    # Start interactive query mode
    try:
        interactive_query_mode()
    finally:
        # Close the Weaviate connection properly
        client.close()
        logger.info("Weaviate connection closed properly")
