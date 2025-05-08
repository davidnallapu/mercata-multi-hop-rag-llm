import os
import logging
import json
from dotenv import load_dotenv
import weaviate
import requests
import re
from openai import OpenAI  # Updated import
import datetime
import time
import flask
from flask import Flask, request, jsonify
from flask_cors import CORS
import random

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
        
        # Implement exponential backoff for rate limits
        max_retries = 5
        base_delay = 1
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=data)
                
                # Check for rate limit error
                if response.status_code == 429:
                    # Parse retry after header if available
                    retry_after = response.headers.get('Retry-After', None)
                    if retry_after and retry_after.isdigit():
                        delay = int(retry_after)
                    else:
                        # Calculate exponential backoff with jitter
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    
                    logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                    
                    # If this is the last attempt, try a fallback model
                    if attempt == max_retries - 1 and model != "gpt-3.5-turbo":
                        logger.warning(f"Falling back to gpt-3.5-turbo after multiple rate limit errors")
                        return query_openai_api_direct(prompt, model="gpt-3.5-turbo")
                    
                    continue
                
                # For all other errors, raise exception
                response.raise_for_status()
                
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            except requests.exceptions.RequestException as e:
                # If this wasn't a rate limit error or we're out of retries, raise it
                if not (hasattr(e.response, 'status_code') and e.response.status_code == 429) or attempt == max_retries - 1:
                    raise
                
                # Otherwise try again with the delay calculated above
        
        # If we've exhausted retries
        return f"API rate limit exceeded after {max_retries} retries. Please try again later."
        
    except Exception as e:
        logger.error(f"Failed to call OpenAI API directly: {e}")
        return f"API call failed: {str(e)}"

def query_openai_api(prompt, model="gpt-4o"):
    """Query OpenAI API using their SDK."""
    logger.info(f"Querying OpenAI API with model: {model}")
    
    try:
        api_key_1 = os.getenv("OPENAI_API_KEY")
        if not api_key_1:
            return "Error: OPENAI_API_KEY not found in environment variables."
        
        # Create client with API key
        client = OpenAI(api_key=api_key_1)
        
        # Implement exponential backoff for rate limits
        max_retries = 5
        base_delay = 1
        for attempt in range(max_retries):
            try:
                # Remove debug prints
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides accurate information about code."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=4000,
                    temperature=0.3
                )
                
                # Return just the content
                return response.choices[0].message.content
            except Exception as e:
                # Check if this is a rate limit error
                if hasattr(e, 'status_code') and e.status_code == 429:
                    # Calculate exponential backoff with jitter
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Rate limit hit. Retrying in {delay:.2f} seconds (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                    
                    # If this is the last attempt, try a fallback model
                    if attempt == max_retries - 1 and model != "gpt-3.5-turbo":
                        logger.warning(f"Falling back to gpt-3.5-turbo after multiple rate limit errors")
                        return query_openai_api(prompt, model="gpt-3.5-turbo")
                else:
                    # For non-rate-limit errors, don't retry
                    raise
        
        # If we've exhausted retries
        return f"API rate limit exceeded after {max_retries} retries. Please try again later."
        
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

def query_with_context(user_query, num_chunks=5, max_tokens=8192, use_direct_api=False, use_openai=True, min_iterations=2, max_iterations=10, verbose=True, openai_model="gpt-4o"):
    """Query language models with context from the Weaviate database with iterative refinement."""
    logger.info(f"Querying with: '{user_query}'")
    
    # Initialize logs list to collect all output
    logs = []
    
    def log(message):
        if verbose:
            print(message)
        logs.append(message)
    
    log(f"\n[QUERY PARAMETERS]")
    log(f"  Query: '{user_query}'")
    log(f"  Model: {openai_model}")
    log(f"  Initial chunks: {num_chunks}")
    log(f"  Max tokens: {max_tokens}")
    log(f"  Min iterations: {min_iterations}")
    log(f"  Max iterations: {max_iterations}")
    log(f"  API mode: {'Direct API' if use_direct_api else 'SDK'}")
    
    try:
        # Initialize chunks as an empty list
        chunks = []
        
        # Extract key terms from the user query for refocusing
        key_terms = re.findall(r'\b\w{4,}\b', user_query.lower())
        key_terms = [term for term in key_terms if term not in ['what', 'when', 'where', 'which', 'does', 'about', 'with', 'this', 'that']]
        
        log(f"\n[KEY TERMS EXTRACTED]")
        log(f"  {', '.join(key_terms)}")
        
        # Get collection for vector search
        collection = client.collections.get("CodeChunk")
        
        # Get initial relevant code chunks from Weaviate - fixed property list
        log(f"\n[PERFORMING INITIAL VECTOR SEARCH]")
        
        vector_results = collection.query.near_text(
            query=user_query,
            limit=num_chunks * 2,  # Increase initial chunk count for better context
            return_properties=["text", "file", "category", "module"]
        )
        
        chunks = vector_results.objects
        
        log(f"  Found {len(chunks)} initial chunks")
        for i, chunk in enumerate(chunks[:3], 1):
            log(f"  - Chunk {i}/{len(chunks)}: {chunk.properties.get('file', 'unknown')} ({len(chunk.properties.get('text', ''))} chars)")
        if len(chunks) > 3:
            log(f"  - ... and {len(chunks) - 3} more chunks")
        
        if not chunks:
            return {"response": "No relevant code found to answer your question.", "logs": logs}
        
        # NEW: Search for language definitions if keywords are mentioned
        keyword_match = re.search(r'keyword[s]?\s+["\']?(\w+)["\']?', user_query, re.IGNORECASE)
        if keyword_match:
            keyword = keyword_match.group(1)
            logger.info(f"Detected keyword search for: {keyword}")
            
            log(f"\n[KEYWORD SEARCH DETECTED]")
            log(f"  Searching for keyword: '{keyword}'")
            
            # First, search for parser files - updated for v4 syntax
            parser_filters = weaviate.classes.query.Filter.by_property("file").like(f"*Parse*") & weaviate.classes.query.Filter.by_property("text").like(f"*{keyword}*")
            
            log(f"  Searching parser files for '{keyword}'...")
            
            parser_results = collection.query.fetch_objects(
                filters=parser_filters,
                limit=3,
                return_properties=["text", "file", "category", "module"]
            )
            
            parser_chunks = parser_results.objects
            
            log(f"  Found {len(parser_chunks)} parser chunks for keyword '{keyword}'")
            
            # Also look in declaration files - updated for v4 syntax
            decl_filters = weaviate.classes.query.Filter.by_property("file").like(f"*Declaration*") & weaviate.classes.query.Filter.by_property("text").like(f"*{keyword}*")
            
            log(f"  Searching declaration files for '{keyword}'...")
            
            decl_results = collection.query.fetch_objects(
                filters=decl_filters,
                limit=3,
                return_properties=["text", "file", "category", "module"]
            )
            
            decl_chunks = decl_results.objects
            
            log(f"  Found {len(decl_chunks)} declaration chunks for keyword '{keyword}'")
            
            # Combine parser and declaration chunks
            language_def_chunks = parser_chunks + decl_chunks
            
            if language_def_chunks:
                logger.info(f"Found {len(language_def_chunks)} language definition chunks for keyword '{keyword}'")
                chunks.extend(language_def_chunks)
                
                log(f"  Total language definition chunks added: {len(language_def_chunks)}")
        
        # Track all retrieved chunks to avoid duplicates
        all_chunk_ids = set([chunk.properties.get('id', hash(chunk.properties.get('text', ''))) for chunk in chunks])
        retrieved_files = set([chunk.properties.get('file', '') for chunk in chunks])
        
        log(f"\n[RETRIEVED FILES]")
        for file in sorted(retrieved_files):
            log(f"  - {file}")
        
        # Get structural information from the knowledge graph
        log(f"\n[QUERYING KNOWLEDGE GRAPH STRUCTURE]")
        
        try:
            entities, relationships = query_codebase_structure(user_query)
            has_structure_info = len(entities) > 0 or len(relationships) > 0
            logger.info(f"Found {len(entities)} entities and {len(relationships)} relationships")
            
            log(f"  Found {len(entities)} entities and {len(relationships)} relationships")
            if entities:
                log(f"\n  [TOP ENTITIES]")
                for i, entity in enumerate(entities[:3], 1):
                    log(f"    {i}. {entity.properties.get('name', 'Unknown')} ({entity.properties.get('type', 'Unknown')})")
            if relationships:
                log(f"\n  [TOP RELATIONSHIPS]")
                for i, rel in enumerate(relationships[:3], 1):
                    log(f"    {i}. {rel.properties.get('source', 'Unknown')} → {rel.properties.get('target', 'Unknown')} ({rel.properties.get('type', 'Unknown')})")
        except Exception as e:
            logger.warning(f"Could not fetch knowledge graph structure: {e}")
            entities = []
            relationships = []
            has_structure_info = False
            
            log(f"  Error fetching knowledge graph: {e}")
        
        # Initial context building - ensure full chunks are provided
        log(f"\n[BUILDING INITIAL CONTEXT]")
        log(f"  Creating context with {len(chunks)} code chunks and {len(entities)} entities")
        
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
        
        # Modify the system instructions to emphasize staying on topic
        system_instructions = (
            "You are an expert Developer Relations (DevRel) professional for STRATO blockchain technology, speaking with "
            "confidence and authority. You're enthusiastic about STRATO's features and capabilities, and you excel at explaining "
            "technical concepts in a clear, practical way. Provide definitive answers using the available code context, "
            "highlighting best practices and practical use cases whenever possible.\n\n"
            
            # Add explicit instruction to stay focused on the original question
            f"IMPORTANT: The user has asked specifically about: '{user_query}'\n"
            f"You MUST stay focused on answering this exact question. Do not get distracted by tangential information.\n"
            f"Key terms in this question: {', '.join(key_terms)}\n\n"
            
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
            "Return your answer in **exactly** with with the following sections(IMPRTANT: AS NEEDED BE CREATIVE): 1. \"Key Components & Purpose\" – table with columns | Component | File / Module | Purpose / Docstring | 2. \"Indexing (or Execution) Flow\" – numbered list of 3‑8 steps describing how the feature works end‑to‑end. 3. \"Best‑practice Notes\" – 2‑5 concise bullets with actionable tips. Do **not** add extra headings or prose outside these sections. Your goal is to help developers successfully implement their solutions with clarity and best practices.\n\n"
            "IMPORTANT SEARCH GUIDANCE:\n"
            "1. If you need more information, make ONE SPECIFIC request at a time using: 'NEED_MORE_INFO: [specific query]'\n"
            "2. Focus on finding SPECIFIC FILES first (e.g., 'Find Declarations.hs file')\n"
            "3. When requesting language definitions, look for parser files (.hs) and declaration files\n" 
            "4. USE ALL AVAILABLE ITERATIONS to gather comprehensive information before answering\n"
            "5. If you mention a file as important but haven't found it yet, your next query MUST search for it\n"
            "6. Do not provide a final answer until you've found ALL critical implementation files you've identified\n"
            "7. MAINTAIN FOCUS on the original question - don't fixate on tangentially related files\n"
            "8. ALWAYS EVALUATE the relevance of files you find to the original question\n\n"
            
            "CRITICAL DATABASE SEARCHING INSTRUCTIONS:\n"
            "1. Your 'NEED_MORE_INFO:' requests will query a vector database directly\n"
            "2. DO NOT use full sentences or questions in your search queries\n"
            "3. USE ONLY specific technical keywords, file names, function names or short phrases\n"
            "4. BAD EXAMPLE: 'NEED_MORE_INFO: Can you find information about how records are parsed in the language?'\n"
            "5. GOOD EXAMPLE: 'NEED_MORE_INFO: record keyword Declaration.hs'\n"
            "6. GOOD EXAMPLE: 'NEED_MORE_INFO: Parser.hs recordDeclaration function'\n"
            "7. GOOD EXAMPLE: 'NEED_MORE_INFO: SolidityParser.sol contract implementation'\n"
            "8. The database doesn't understand natural language - use ONLY specific technical keywords!\n"
            "9. KEEP QUERIES UNDER 10 WORDS for best results\n\n"
            
            "Good search examples:\n"
            "- 'NEED_MORE_INFO: Declarations.hs parser file'\n"
            "- 'NEED_MORE_INFO: keyword definitions Parse directory'\n"
            "- 'NEED_MORE_INFO: record keyword implementation SolidVM'\n"
            "- 'NEED_MORE_INFO: language syntax definitions SolidVM'\n"
            "- 'NEED_MORE_INFO: GraphRelationship function X components'\n\n"
            
            "RESPONSE FORMAT REQUIREMENTS:\n"
            "1. ALWAYS include relevant code chunks in your response\n"
            "2. When referring to code, cite the specific CHUNK number\n"
            "3. Provide detailed explanations with specific function names, parameters, and return values\n"
            "4. Explain how the code works with concrete examples\n"
            "5. Your response should be detailed and comprehensive - at least 3000 words\n"
            "6. Break down complex relationships between components\n"
            "7. Always show implementation details from the code chunks\n\n"
            "8. MAKE SURE YOU INCLUDE ALL RELEVANT CODE CHUNKS IN YOUR RESPONSE\n"
            "9. GIVE DETAILED RESPONSE, MINIMUM 3000 OR MORE WORDS AND 5-10 SECTIONS\n"
            "10. INCLUDE EXAMPLES WHENEVER POSSIBLE\n"
        )
        
        log(f"\n[SYSTEM INSTRUCTIONS LENGTH]")
        log(f"  {len(system_instructions)} characters")
        
        initial_prompt = f"{system_instructions}\n\n{context}\n\nBased on the code above, please answer this question with DevRel confidence and expertise: {user_query}\n\n" + \
                         f"Retrieved files so far: {', '.join(retrieved_files)}"
        
        log(f"\n[INITIAL PROMPT OVERVIEW]")
        log(f"  Total size: {len(initial_prompt)} characters")
        log(f"  System instructions: {len(system_instructions)} characters")
        log(f"  Context: {len(context)} characters")
        total_tokens = len(initial_prompt) / 4  # Very rough estimate
        log(f"  Estimated tokens: ~{int(total_tokens)} (rough estimate)")
        
        conversation_history = []
        iterations = 0
        final_answer = None
        
        # Begin iterative process
        while iterations < max_iterations:
            iterations += 1
            logger.info(f"Starting iteration {iterations}/{max_iterations}")
            
            log(f"\n[ITERATION {iterations}/{max_iterations}]")
            log(f"  Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Get LLM response
            if use_openai:
                current_prompt = initial_prompt if iterations == 1 else f"{initial_prompt}\n\nConversation so far:\n{''.join(conversation_history)}\n\nContinue helping the user with DevRel expertise and confidence."
                
                # For early iterations, explicitly instruct the LLM to keep searching
                if iterations < min_iterations:
                    current_prompt += f"\n\nIMPORTANT: This is iteration {iterations} of a minimum of {min_iterations} required iterations. DO NOT provide a final answer yet. Instead, use 'NEED_MORE_INFO:' to request additional specific information about the codebase to build a more complete understanding."
                
                log(f"  Sending request to OpenAI API (model: {openai_model})")
                log(f"  Prompt length: {len(current_prompt)} characters")
                if iterations > 1:
                    log(f"  Conversation history: {len(conversation_history)} exchanges")
                
                start_time = time.time()
                response = query_openai_api(current_prompt, model=openai_model)
                end_time = time.time()
                
                log(f"  API call completed in {end_time - start_time:.2f} seconds")
                log(f"  Response length: {len(response)} characters")
            elif use_direct_api:
                current_prompt = initial_prompt if iterations == 1 else f"{initial_prompt}\n\nConversation so far:\n{''.join(conversation_history)}\n\nContinue helping the user with DevRel expertise and confidence."
                
                log(f"  Sending direct request to OpenAI API (model: {openai_model})")
                log(f"  Using direct API call method")
                log(f"  Prompt length: {len(current_prompt)} characters")
                
                start_time = time.time()
                response = query_openai_api_direct(current_prompt, model=openai_model)
                end_time = time.time()
                
                log(f"  API call completed in {end_time - start_time:.2f} seconds")
                log(f"  Response length: {len(response)} characters")
            else:
                # Fallback to using the OpenAI SDK
                try:
                    current_prompt = initial_prompt if iterations == 1 else f"{initial_prompt}\n\nConversation so far:\n{''.join(conversation_history)}\n\nContinue helping the user with DevRel expertise and confidence."
                    api_key_1 = os.getenv("OPENAI_API_KEY")
                    if not api_key_1:
                        return {"response": "Error: OPENAI_API_KEY not found in environment variables.", "logs": logs}
                    
                    log(f"  Using fallback OpenAI SDK method")
                    log(f"  Prompt length: {len(current_prompt)} characters")
                    
                    # Create client with API key - use a different name to avoid conflict
                    openai_client = OpenAI(api_key=api_key_1)

                    start_time = time.time()
                    response = openai_client.chat.completions.create(
                        model=openai_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that provides accurate information about code."},
                            {"role": "user", "content": current_prompt}
                        ],
                        max_tokens=4000,
                        temperature=0.3
                    ).choices[0].message.content
                    end_time = time.time()
                    
                    log(f"  API call completed in {end_time - start_time:.2f} seconds")
                    log(f"  Response length: {len(response)} characters")
                except Exception as e:
                    logger.error(f"Failed to call OpenAI API: {e}")
                    log(f"  ERROR: Failed to call OpenAI API: {e}")
                    return {"response": f"LLM failed: {str(e)}", "logs": logs}
            
            # Check if LLM needs more information
            if "NEED_MORE_INFO:" in response:
                follow_up_query = response.split("NEED_MORE_INFO:")[1].strip()
                follow_up_query = follow_up_query.split("\n")[0].strip()  # Get just the first line
                
                log(f"\n[LLM REQUESTING MORE INFORMATION]")
                log(f"  Query: '{follow_up_query}'")
                
                # NEW: Check if search is drifting off-topic and refocus if needed
                contains_key_terms = any(term in follow_up_query.lower() for term in key_terms)
                if not contains_key_terms and iterations > 1:
                    # Refocus the search by adding key terms from original question
                    original_terms = " ".join(key_terms[:3])  # Use first few key terms
                    follow_up_query = f"{follow_up_query} {original_terms}"
                    logger.info(f"Refocusing search with original terms: {follow_up_query}")
                    
                    log(f"  REFOCUSING: Adding original key terms to stay on topic")
                    log(f"  New query: '{follow_up_query}'")
                
                logger.info(f"LLM requesting more info: {follow_up_query}")
                
                # Log the interaction
                conversation_history.append(f"\nLLM: {response}\n")
                
                # NEW: Check if the query is specifically looking for a file
                file_pattern = re.compile(r'([\w.-]+\.(sol|hs|js|jsx|ts|tsx|py))', re.IGNORECASE)
                file_matches = file_pattern.findall(follow_up_query)
                
                if file_matches:
                    # This appears to be a file search
                    log(f"\n[FILE SEARCH DETECTED]")
                    log(f"  Searching for specific files: {[match[0] for match in file_matches]}")
                    
                    for file_match, _ in file_matches:
                        logger.info(f"Detected file search for: {file_match}")
                        
                        log(f"  Searching for file: '{file_match}'")
                        
                        # Create a file filter query - find files that contain this filename
                        file_filter = weaviate.classes.query.Filter.by_property("file").like(f"*{file_match}*")
                        
                        start_time = time.time()
                        file_results = collection.query.fetch_objects(
                            filters=file_filter,
                            limit=num_chunks,
                            return_properties=["text", "file", "category", "module"]
                        )
                        end_time = time.time()
                        
                        new_chunks = file_results.objects
                        
                        log(f"  Search completed in {end_time - start_time:.2f} seconds")
                        log(f"  Found {len(new_chunks)} chunks from files matching '{file_match}'")
                        
                        if new_chunks:
                            log(f"  Files found:")
                            files_found = set([chunk.properties.get('file', '') for chunk in new_chunks])
                            for file in files_found:
                                log(f"    - {file}")
                            
                            # Filter out duplicates
                            unique_new_chunks = []
                            for chunk in new_chunks:
                                chunk_id = chunk.properties.get('id', hash(chunk.properties.get('text', '')))
                                if chunk_id not in all_chunk_ids:
                                    all_chunk_ids.add(chunk_id)
                                    unique_new_chunks.append(chunk)
                                    retrieved_files.add(chunk.properties.get('file', ''))
                            
                            log(f"  Unique new chunks: {len(unique_new_chunks)} (after filtering duplicates)")
                            
                            # Add these chunks to our ongoing list
                            if unique_new_chunks:
                                additional_context = "\nADDITIONAL FILE CHUNKS FOUND:\n\n"
                                for i, chunk in enumerate(unique_new_chunks, len(chunks)+1):
                                    file_path = chunk.properties.get('file', 'unknown')
                                    category = chunk.properties.get('category', 'unknown')
                                    additional_context += f"CHUNK {i} (from {file_path}, category: {category}):\n```\n{chunk.properties.get('text', '')}\n```\n\n"
                                
                                # Add to conversation history and extend chunks
                                conversation_history.append(f"\nSYSTEM: {additional_context}\n")
                                chunks.extend(unique_new_chunks)
                                
                                log(f"  Added {len(unique_new_chunks)} new unique chunks to context")
                                
                                # Skip the semantic search if we found file-specific results
                                continue
                
                # Perform the regular semantic search
                log(f"\n[PERFORMING SEMANTIC SEARCH]")
                log(f"  Query: '{follow_up_query}'")
                
                start_time = time.time()
                new_vector_results = collection.query.near_text(
                    query=follow_up_query,
                    limit=num_chunks,
                    return_properties=["text", "file", "category", "module"]
                )
                end_time = time.time()
                
                new_chunks = new_vector_results.objects
                
                log(f"  Search completed in {end_time - start_time:.2f} seconds")
                log(f"  Found {len(new_chunks)} chunks")
                
                # Filter out duplicates
                unique_new_chunks = []
                for chunk in new_chunks:
                    chunk_id = chunk.properties.get('id', hash(chunk.properties.get('text', '')))
                    if chunk_id not in all_chunk_ids:
                        all_chunk_ids.add(chunk_id)
                        unique_new_chunks.append(chunk)
                        retrieved_files.add(chunk.properties.get('file', ''))
                
                log(f"  Unique new chunks: {len(unique_new_chunks)} (after filtering duplicates)")
                if unique_new_chunks:
                    log(f"  New files found:")
                    new_files = set([chunk.properties.get('file', '') for chunk in unique_new_chunks])
                    for file in new_files:
                        log(f"    - {file}")
                
                # Add new chunks to context
                if unique_new_chunks:
                    additional_context = "\nADDITIONAL CHUNKS FOUND:\n\n"
                    for i, chunk in enumerate(unique_new_chunks, len(chunks)+1):
                        file_path = chunk.properties.get('file', 'unknown')
                        category = chunk.properties.get('category', 'unknown')
                        additional_context += f"CHUNK {i} (from {file_path}, category: {category}):\n```\n{chunk.properties.get('text', '')}\n```\n\n"
                    
                    # Add to conversation history
                    conversation_history.append(f"\nSYSTEM: {additional_context}\n")
                    chunks.extend(unique_new_chunks)
                    
                    log(f"  Added {len(unique_new_chunks)} new unique chunks to context")
                    log(f"  Total chunks now: {len(chunks)}")
                    log(f"  Files retrieved so far: {len(retrieved_files)}")
                else:
                    conversation_history.append("\nSYSTEM: No additional relevant chunks found.\n")
                    log("  No additional relevant code chunks found")
            else:
                # LLM has provided a final answer - but check if we've met minimum iterations
                if iterations >= min_iterations:
                    log(f"\n[FINAL ANSWER PROVIDED]")
                    log(f"  After {iterations} iterations (minimum required: {min_iterations})")
                    log(f"  Response length: {len(response)} characters")
                    
                    final_answer = response
                    # Exit the loop once we have a final answer AND have met minimum iterations
                    break
                else:
                    # Force continued exploration by converting the response into a NEED_MORE_INFO request
                    logger.info(f"LLM tried to answer early at iteration {iterations}/{min_iterations}. Forcing continued exploration.")
                    log(f"\n[FORCING CONTINUED EXPLORATION]")
                    log(f"  AI tried to answer at iteration {iterations} but minimum is {min_iterations}")
                    
                    # Extract a follow-up query from the response or create a general one
                    follow_up_query = f"Find more information about {user_query}"
                    
                    log(f"  Generated follow-up query: '{follow_up_query}'")
                    
                    # Add forced exploration message to conversation history
                    conversation_history.append(f"\nLLM: {response}\n")
                    conversation_history.append(f"\nSYSTEM: You must continue exploring for at least {min_iterations} iterations. Currently at iteration {iterations}. Please find more relevant information.\n")
                    
                    # Get additional chunks with this follow-up query
                    log(f"\n[FORCING ADDITIONAL SEARCH]")
                    
                    start_time = time.time()
                    new_vector_results = collection.query.near_text(
                        query=follow_up_query,
                        limit=num_chunks,
                        return_properties=["text", "file", "category", "module"]
                    )
                    end_time = time.time()
                    
                    new_chunks = new_vector_results.objects
                    
                    log(f"  Search completed in {end_time - start_time:.2f} seconds")
                    log(f"  Found {len(new_chunks)} chunks")
                    
                    # Filter out duplicates
                    unique_new_chunks = []
                    for chunk in new_chunks:
                        chunk_id = chunk.properties.get('id', hash(chunk.properties.get('text', '')))
                        if chunk_id not in all_chunk_ids:
                            all_chunk_ids.add(chunk_id)
                            unique_new_chunks.append(chunk)
                            retrieved_files.add(chunk.properties.get('file', ''))
                    
                    log(f"  Unique new chunks: {len(unique_new_chunks)} (after filtering duplicates)")
                    
                    # Add new chunks to context
                    if unique_new_chunks:
                        additional_context = "\nADDITIONAL CHUNKS FOUND:\n\n"
                        for i, chunk in enumerate(unique_new_chunks, len(chunks)+1):
                            file_path = chunk.properties.get('file', 'unknown')
                            category = chunk.properties.get('category', 'unknown')
                            additional_context += f"CHUNK {i} (from {file_path}, category: {category}):\n```\n{chunk.properties.get('text', '')}\n```\n\n"
                        
                        # Add to conversation history
                        conversation_history.append(f"\nSYSTEM: {additional_context}\n")
                        chunks.extend(unique_new_chunks)
                        
                        log(f"  Added {len(unique_new_chunks)} new unique chunks to context")
                        log(f"  Total chunks now: {len(chunks)}")
                        log(f"  Files retrieved so far: {len(retrieved_files)}")
                    else:
                        conversation_history.append("\nSYSTEM: No additional relevant chunks found.\n")
                        log("  No additional relevant code chunks found")
            
            # Add refocusing reminder after each iteration
            if iterations > 1:
                refocus_message = f"\nSYSTEM REMINDER: The original question was '{user_query}'. Stay focused on this question.\n"
                conversation_history.append(refocus_message)
                
                log(f"\n[ADDED REFOCUSING REMINDER]")
            
            # Provide iteration summary if verbose
            log(f"\n[ITERATION {iterations} COMPLETE]")
            log(f"  Total chunks gathered: {len(chunks)}")
            log(f"  Total files retrieved: {len(retrieved_files)}")
            log(f"  Conversation history size: {len(''.join(conversation_history))} characters")
            log(f"  Moving to iteration {iterations+1} of max {max_iterations}")
            
            # At the end of each iteration
            yield {"type": "iteration_complete", "iteration": iterations, "max_iterations": max_iterations}
        
        # Final result
        yield {"type": "result", "response": final_answer if final_answer else response, "logs": logs}
        
    except Exception as e:
        logger.error(f"Error during streaming query: {e}")
        yield from log(f"\n[ERROR OCCURRED]")
        yield from log(f"  {str(e)}")
        import traceback
        yield from log(f"  Traceback: {traceback.format_exc()}")
        yield {"type": "error", "message": str(e), "logs": logs}

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
    print("Type 'min N' to set min iteration depth (default: 5)")
    print("Type 'verbose on/off' to toggle detailed progress (default: on)")
    print("Type 'chunks N' to set number of chunks to retrieve (default: 5)")
    
    use_direct_api = False
    use_openai = True
    openai_model = "gpt-4o"
    max_iterations = 10
    min_iterations = 2
    verbose = True
    num_chunks = 10
    
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
                print("Invalid min value. Using default of 2.")
                min_iterations = 2
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
                print("Invalid chunks value. Using default of 10.")
                num_chunks = 10
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

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/query', methods=['POST'])
def process_query():
    """Handle POST requests to the /query endpoint."""
    try:
        # Get JSON data from request
        data = request.json
        
        if not data or 'query' not in data:
            return jsonify({'error': 'Missing query parameter'}), 400
        
        query_text = data.get('query')
        
        # Optional parameters
        num_chunks = data.get('num_chunks', 10)
        use_direct_api = data.get('use_direct_api', False)
        use_openai = data.get('use_openai', True)
        min_iterations = data.get('min_iterations', 2)
        max_iterations = data.get('max_iterations', 10)
        verbose = data.get('verbose', True)
        openai_model = data.get('model', 'gpt-4o')
        
        # Check if streaming is requested
        stream_mode = data.get('stream', False)
        
        if stream_mode:
            # Create a streaming response
            def generate():
                for update in query_with_context_stream(
                    query_text,
                    num_chunks=num_chunks,
                    use_direct_api=use_direct_api,
                    use_openai=use_openai,
                    min_iterations=min_iterations,
                    max_iterations=max_iterations,
                    verbose=verbose,
                    openai_model=openai_model
                ):
                    yield json.dumps(update) + "\n"
            
            return flask.Response(generate(), mimetype='application/x-ndjson')
        else:
            # Execute the query and get the response with logs (non-streaming)
            result = query_with_context(
                query_text,
                num_chunks=num_chunks,
                use_direct_api=use_direct_api,
                use_openai=use_openai,
                min_iterations=min_iterations,
                max_iterations=max_iterations,
                verbose=verbose,
                openai_model=openai_model
            )
            
            # Return response as JSON
            return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

def query_with_context_stream(user_query, num_chunks=5, max_tokens=8192, use_direct_api=False, use_openai=True, min_iterations=2, max_iterations=10, verbose=True, openai_model="gpt-4o"):
    """Streaming version of query_with_context that yields updates as they happen."""
    logger.info(f"Streaming query with: '{user_query}'")
    
    # Initialize logs list to collect all output
    logs = []
    
    def log(message):
        if verbose:
            print(message)
        logs.append(message)
        # Yield log updates for streaming
        yield {"type": "log", "message": message, "logs": logs}
    
    # Initial update with query parameters
    yield {"type": "start", "query": user_query, "parameters": {
        "model": openai_model,
        "chunks": num_chunks,
        "max_tokens": max_tokens,
        "min_iterations": min_iterations,
        "max_iterations": max_iterations,
        "api_mode": 'Direct API' if use_direct_api else 'SDK'
    }}
    
    yield from log(f"\n[QUERY PARAMETERS]")
    yield from log(f"  Query: '{user_query}'")
    yield from log(f"  Model: {openai_model}")
    yield from log(f"  Initial chunks: {num_chunks}")
    yield from log(f"  Max tokens: {max_tokens}")
    yield from log(f"  Min iterations: {min_iterations}")
    yield from log(f"  Max iterations: {max_iterations}")
    yield from log(f"  API mode: {'Direct API' if use_direct_api else 'SDK'}")
    
    try:
        # Initialize chunks as an empty list
        chunks = []
        
        # Extract key terms from the user query for refocusing
        key_terms = re.findall(r'\b\w{4,}\b', user_query.lower())
        key_terms = [term for term in key_terms if term not in ['what', 'when', 'where', 'which', 'does', 'about', 'with', 'this', 'that']]
        
        yield from log(f"\n[KEY TERMS EXTRACTED]")
        yield from log(f"  {', '.join(key_terms)}")
        
        # Get collection for vector search
        collection = client.collections.get("CodeChunk")
        
        # Get initial relevant code chunks from Weaviate - fixed property list
        yield from log(f"\n[PERFORMING INITIAL VECTOR SEARCH]")
        
        vector_results = collection.query.near_text(
            query=user_query,
            limit=num_chunks * 2,  # Increase initial chunk count for better context
            return_properties=["text", "file", "category", "module"]
        )
        
        chunks = vector_results.objects
        
        yield from log(f"  Found {len(chunks)} initial chunks")
        for i, chunk in enumerate(chunks[:3], 1):
            yield from log(f"  - Chunk {i}/{len(chunks)}: {chunk.properties.get('file', 'unknown')} ({len(chunk.properties.get('text', ''))} chars)")
        if len(chunks) > 3:
            yield from log(f"  - ... and {len(chunks) - 3} more chunks")
        
        if not chunks:
            yield {"type": "result", "response": "No relevant code found to answer your question.", "logs": logs}
            return

        # Track all retrieved chunks to avoid duplicates
        all_chunk_ids = set([chunk.properties.get('id', hash(chunk.properties.get('text', ''))) for chunk in chunks])
        retrieved_files = set([chunk.properties.get('file', '') for chunk in chunks])
        
        yield from log(f"\n[RETRIEVED FILES]")
        for file in sorted(retrieved_files):
            yield from log(f"  - {file}")
        
        # Get structural information from the knowledge graph
        yield from log(f"\n[QUERYING KNOWLEDGE GRAPH STRUCTURE]")
        
        try:
            entities, relationships = query_codebase_structure(user_query)
            has_structure_info = len(entities) > 0 or len(relationships) > 0
            logger.info(f"Found {len(entities)} entities and {len(relationships)} relationships")
            
            yield from log(f"  Found {len(entities)} entities and {len(relationships)} relationships")
            if entities:
                yield from log(f"\n  [TOP ENTITIES]")
                for i, entity in enumerate(entities[:3], 1):
                    yield from log(f"    {i}. {entity.properties.get('name', 'Unknown')} ({entity.properties.get('type', 'Unknown')})")
            if relationships:
                yield from log(f"\n  [TOP RELATIONSHIPS]")
                for i, rel in enumerate(relationships[:3], 1):
                    yield from log(f"    {i}. {rel.properties.get('source', 'Unknown')} → {rel.properties.get('target', 'Unknown')} ({rel.properties.get('type', 'Unknown')})")
        except Exception as e:
            logger.warning(f"Could not fetch knowledge graph structure: {e}")
            entities = []
            relationships = []
            has_structure_info = False
            
            yield from log(f"  Error fetching knowledge graph: {e}")
        
        # Initial context building - ensure full chunks are provided
        yield from log(f"\n[BUILDING INITIAL CONTEXT]")
        yield from log(f"  Creating context with {len(chunks)} code chunks and {len(entities)} entities")
        
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
        
        # Modify the system instructions to emphasize staying on topic
        system_instructions = (
            "You are an expert Developer Relations (DevRel) professional for STRATO blockchain technology, speaking with "
            "confidence and authority. You're enthusiastic about STRATO's features and capabilities, and you excel at explaining "
            "technical concepts in a clear, practical way. Provide definitive answers using the available code context, "
            "highlighting best practices and practical use cases whenever possible.\n\n"
            
            # Add explicit instruction to stay focused on the original question
            f"IMPORTANT: The user has asked specifically about: '{user_query}'\n"
            f"You MUST stay focused on answering this exact question. Do not get distracted by tangential information.\n"
            f"Key terms in this question: {', '.join(key_terms)}\n\n"
            
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
            "Return your answer in **exactly** with with the following sections(IMPRTANT: AS NEEDED BE CREATIVE): 1. \"Key Components & Purpose\" – table with columns | Component | File / Module | Purpose / Docstring | 2. \"Indexing (or Execution) Flow\" – numbered list of 3‑8 steps describing how the feature works end‑to‑end. 3. \"Best‑practice Notes\" – 2‑5 concise bullets with actionable tips. Do **not** add extra headings or prose outside these sections. Your goal is to help developers successfully implement their solutions with clarity and best practices.\n\n"
            "IMPORTANT SEARCH GUIDANCE:\n"
            "1. If you need more information, make ONE SPECIFIC request at a time using: 'NEED_MORE_INFO: [specific query]'\n"
            "2. Focus on finding SPECIFIC FILES first (e.g., 'Find Declarations.hs file')\n"
            "3. When requesting language definitions, look for parser files (.hs) and declaration files\n" 
            "4. USE ALL AVAILABLE ITERATIONS to gather comprehensive information before answering\n"
            "5. If you mention a file as important but haven't found it yet, your next query MUST search for it\n"
            "6. Do not provide a final answer until you've found ALL critical implementation files you've identified\n"
            "7. MAINTAIN FOCUS on the original question - don't fixate on tangentially related files\n"
            "8. ALWAYS EVALUATE the relevance of files you find to the original question\n\n"
            
            "CRITICAL DATABASE SEARCHING INSTRUCTIONS:\n"
            "1. Your 'NEED_MORE_INFO:' requests will query a vector database directly\n"
            "2. DO NOT use full sentences or questions in your search queries\n"
            "3. USE ONLY specific technical keywords, file names, function names or short phrases\n"
            "4. BAD EXAMPLE: 'NEED_MORE_INFO: Can you find information about how records are parsed in the language?'\n"
            "5. GOOD EXAMPLE: 'NEED_MORE_INFO: record keyword Declaration.hs'\n"
            "6. GOOD EXAMPLE: 'NEED_MORE_INFO: Parser.hs recordDeclaration function'\n"
            "7. GOOD EXAMPLE: 'NEED_MORE_INFO: SolidityParser.sol contract implementation'\n"
            "8. The database doesn't understand natural language - use ONLY specific technical keywords!\n"
            "9. KEEP QUERIES UNDER 10 WORDS for best results\n\n"
            
            "Good search examples:\n"
            "- 'NEED_MORE_INFO: Declarations.hs parser file'\n"
            "- 'NEED_MORE_INFO: keyword definitions Parse directory'\n"
            "- 'NEED_MORE_INFO: record keyword implementation SolidVM'\n"
            "- 'NEED_MORE_INFO: language syntax definitions SolidVM'\n"
            "- 'NEED_MORE_INFO: GraphRelationship function X components'\n\n"
            
            "RESPONSE FORMAT REQUIREMENTS:\n"
            "1. ALWAYS include relevant code chunks in your response\n"
            "2. When referring to code, cite the specific CHUNK number\n"
            "3. Provide detailed explanations with specific function names, parameters, and return values\n"
            "4. Explain how the code works with concrete examples\n"
            "5. Your response should be detailed and comprehensive - at least 3000 words\n"
            "6. Break down complex relationships between components\n"
            "7. Always show implementation details from the code chunks\n\n"
            "8. MAKE SURE YOU INCLUDE ALL RELEVANT CODE CHUNKS IN YOUR RESPONSE\n"
            "9. GIVE DETAILED RESPONSE, MINIMUM 3000 OR MORE WORDS AND 5-10 SECTIONS\n"
            "10. INCLUDE EXAMPLES WHENEVER POSSIBLE\n"
        )
        
        yield from log(f"\n[SYSTEM INSTRUCTIONS LENGTH]")
        yield from log(f"  {len(system_instructions)} characters")
        
        initial_prompt = f"{system_instructions}\n\n{context}\n\nBased on the code above, please answer this question with DevRel confidence and expertise: {user_query}\n\n" + \
                         f"Retrieved files so far: {', '.join(retrieved_files)}"
        
        yield from log(f"\n[INITIAL PROMPT OVERVIEW]")
        yield from log(f"  Total size: {len(initial_prompt)} characters")
        yield from log(f"  System instructions: {len(system_instructions)} characters")
        yield from log(f"  Context: {len(context)} characters")
        total_tokens = len(initial_prompt) / 4  # Very rough estimate
        yield from log(f"  Estimated tokens: ~{int(total_tokens)} (rough estimate)")
        
        conversation_history = []
        iterations = 0
        final_answer = None
        
        # Begin iterative process
        while iterations < max_iterations:
            iterations += 1
            logger.info(f"Starting iteration {iterations}/{max_iterations}")
            
            yield from log(f"\n[ITERATION {iterations}/{max_iterations}]")
            yield from log(f"  Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Yield iteration start event
            yield {"type": "iteration_start", "iteration": iterations, "max_iterations": max_iterations}
            
            # Get LLM response
            if use_openai:
                current_prompt = initial_prompt if iterations == 1 else f"{initial_prompt}\n\nConversation so far:\n{''.join(conversation_history)}\n\nContinue helping the user with DevRel expertise and confidence."
                
                # For early iterations, explicitly instruct the LLM to keep searching
                if iterations < min_iterations:
                    current_prompt += f"\n\nIMPORTANT: This is iteration {iterations} of a minimum of {min_iterations} required iterations. DO NOT provide a final answer yet. Instead, use 'NEED_MORE_INFO:' to request additional specific information about the codebase to build a more complete understanding."
                
                log(f"  Sending request to OpenAI API (model: {openai_model})")
                log(f"  Prompt length: {len(current_prompt)} characters")
                if iterations > 1:
                    log(f"  Conversation history: {len(conversation_history)} exchanges")
                
                start_time = time.time()
                response = query_openai_api(current_prompt, model=openai_model)
                end_time = time.time()
                
                log(f"  API call completed in {end_time - start_time:.2f} seconds")
                log(f"  Response length: {len(response)} characters")
                
                # After getting the response, yield it
                yield {"type": "llm_response", "iteration": iterations, "response": response}
            elif use_direct_api:
                current_prompt = initial_prompt if iterations == 1 else f"{initial_prompt}\n\nConversation so far:\n{''.join(conversation_history)}\n\nContinue helping the user with DevRel expertise and confidence."
                
                log(f"  Sending direct request to OpenAI API (model: {openai_model})")
                log(f"  Using direct API call method")
                log(f"  Prompt length: {len(current_prompt)} characters")
                
                start_time = time.time()
                response = query_openai_api_direct(current_prompt, model=openai_model)
                end_time = time.time()
                
                log(f"  API call completed in {end_time - start_time:.2f} seconds")
                log(f"  Response length: {len(response)} characters")
                
                # After getting the response, yield it
                yield {"type": "llm_response", "iteration": iterations, "response": response}
            else:
                # Fallback to using the OpenAI SDK
                try:
                    current_prompt = initial_prompt if iterations == 1 else f"{initial_prompt}\n\nConversation so far:\n{''.join(conversation_history)}\n\nContinue helping the user with DevRel expertise and confidence."
                    api_key_1 = os.getenv("OPENAI_API_KEY")
                    if not api_key_1:
                        yield {"type": "error", "message": "Error: OPENAI_API_KEY not found in environment variables.", "logs": logs}
                        return
                    
                    log(f"  Using fallback OpenAI SDK method")
                    log(f"  Prompt length: {len(current_prompt)} characters")
                    
                    # Create client with API key - use a different name to avoid conflict
                    openai_client = OpenAI(api_key=api_key_1)

                    start_time = time.time()
                    response = openai_client.chat.completions.create(
                        model=openai_model,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that provides accurate information about code."},
                            {"role": "user", "content": current_prompt}
                        ],
                        max_tokens=4000,
                        temperature=0.3
                    ).choices[0].message.content
                    end_time = time.time()
                    
                    log(f"  API call completed in {end_time - start_time:.2f} seconds")
                    log(f"  Response length: {len(response)} characters")
                    
                    # After getting the response, yield it
                    yield {"type": "llm_response", "iteration": iterations, "response": response}
                except Exception as e:
                    logger.error(f"Failed to call OpenAI API: {e}")
                    log(f"  ERROR: Failed to call OpenAI API: {e}")
                    yield {"type": "error", "message": f"LLM failed: {str(e)}", "logs": logs}
                    return
            
            # Check if LLM needs more information
            if "NEED_MORE_INFO:" in response:
                follow_up_query = response.split("NEED_MORE_INFO:")[1].strip()
                follow_up_query = follow_up_query.split("\n")[0].strip()  # Get just the first line
                
                log(f"\n[LLM REQUESTING MORE INFORMATION]")
                log(f"  Query: '{follow_up_query}'")
                
                # NEW: Check if search is drifting off-topic and refocus if needed
                contains_key_terms = any(term in follow_up_query.lower() for term in key_terms)
                if not contains_key_terms and iterations > 1:
                    # Refocus the search by adding key terms from original question
                    original_terms = " ".join(key_terms[:3])  # Use first few key terms
                    follow_up_query = f"{follow_up_query} {original_terms}"
                    logger.info(f"Refocusing search with original terms: {follow_up_query}")
                    
                    log(f"  REFOCUSING: Adding original key terms to stay on topic")
                    log(f"  New query: '{follow_up_query}'")
                
                logger.info(f"LLM requesting more info: {follow_up_query}")
                
                # Log the interaction
                conversation_history.append(f"\nLLM: {response}\n")
                
                # NEW: Check if the query is specifically looking for a file
                file_pattern = re.compile(r'([\w.-]+\.(sol|hs|js|jsx|ts|tsx|py))', re.IGNORECASE)
                file_matches = file_pattern.findall(follow_up_query)
                
                if file_matches:
                    # This appears to be a file search
                    log(f"\n[FILE SEARCH DETECTED]")
                    log(f"  Searching for specific files: {[match[0] for match in file_matches]}")
                    
                    for file_match, _ in file_matches:
                        logger.info(f"Detected file search for: {file_match}")
                        
                        log(f"  Searching for file: '{file_match}'")
                        
                        # Create a file filter query - find files that contain this filename
                        file_filter = weaviate.classes.query.Filter.by_property("file").like(f"*{file_match}*")
                        
                        start_time = time.time()
                        file_results = collection.query.fetch_objects(
                            filters=file_filter,
                            limit=num_chunks,
                            return_properties=["text", "file", "category", "module"]
                        )
                        end_time = time.time()
                        
                        new_chunks = file_results.objects
                        
                        log(f"  Search completed in {end_time - start_time:.2f} seconds")
                        log(f"  Found {len(new_chunks)} chunks from files matching '{file_match}'")
                        
                        if new_chunks:
                            log(f"  Files found:")
                            files_found = set([chunk.properties.get('file', '') for chunk in new_chunks])
                            for file in files_found:
                                log(f"    - {file}")
                            
                            # Filter out duplicates
                            unique_new_chunks = []
                            for chunk in new_chunks:
                                chunk_id = chunk.properties.get('id', hash(chunk.properties.get('text', '')))
                                if chunk_id not in all_chunk_ids:
                                    all_chunk_ids.add(chunk_id)
                                    unique_new_chunks.append(chunk)
                                    retrieved_files.add(chunk.properties.get('file', ''))
                            
                            log(f"  Unique new chunks: {len(unique_new_chunks)} (after filtering duplicates)")
                            
                            # Add these chunks to our ongoing list
                            if unique_new_chunks:
                                additional_context = "\nADDITIONAL FILE CHUNKS FOUND:\n\n"
                                for i, chunk in enumerate(unique_new_chunks, len(chunks)+1):
                                    file_path = chunk.properties.get('file', 'unknown')
                                    category = chunk.properties.get('category', 'unknown')
                                    additional_context += f"CHUNK {i} (from {file_path}, category: {category}):\n```\n{chunk.properties.get('text', '')}\n```\n\n"
                                
                                # Add to conversation history and extend chunks
                                conversation_history.append(f"\nSYSTEM: {additional_context}\n")
                                chunks.extend(unique_new_chunks)
                                
                                log(f"  Added {len(unique_new_chunks)} new unique chunks to context")
                                
                                # Skip the semantic search if we found file-specific results
                                continue
                
                # Perform the regular semantic search
                log(f"\n[PERFORMING SEMANTIC SEARCH]")
                log(f"  Query: '{follow_up_query}'")
                
                start_time = time.time()
                new_vector_results = collection.query.near_text(
                    query=follow_up_query,
                    limit=num_chunks,
                    return_properties=["text", "file", "category", "module"]
                )
                end_time = time.time()
                
                new_chunks = new_vector_results.objects
                
                log(f"  Search completed in {end_time - start_time:.2f} seconds")
                log(f"  Found {len(new_chunks)} chunks")
                
                # Filter out duplicates
                unique_new_chunks = []
                for chunk in new_chunks:
                    chunk_id = chunk.properties.get('id', hash(chunk.properties.get('text', '')))
                    if chunk_id not in all_chunk_ids:
                        all_chunk_ids.add(chunk_id)
                        unique_new_chunks.append(chunk)
                        retrieved_files.add(chunk.properties.get('file', ''))
                
                log(f"  Unique new chunks: {len(unique_new_chunks)} (after filtering duplicates)")
                if unique_new_chunks:
                    log(f"  New files found:")
                    new_files = set([chunk.properties.get('file', '') for chunk in unique_new_chunks])
                    for file in new_files:
                        log(f"    - {file}")
                
                # Add new chunks to context
                if unique_new_chunks:
                    additional_context = "\nADDITIONAL CHUNKS FOUND:\n\n"
                    for i, chunk in enumerate(unique_new_chunks, len(chunks)+1):
                        file_path = chunk.properties.get('file', 'unknown')
                        category = chunk.properties.get('category', 'unknown')
                        additional_context += f"CHUNK {i} (from {file_path}, category: {category}):\n```\n{chunk.properties.get('text', '')}\n```\n\n"
                    
                    # Add to conversation history
                    conversation_history.append(f"\nSYSTEM: {additional_context}\n")
                    chunks.extend(unique_new_chunks)
                    
                    log(f"  Added {len(unique_new_chunks)} new unique chunks to context")
                    log(f"  Total chunks now: {len(chunks)}")
                    log(f"  Files retrieved so far: {len(retrieved_files)}")
                else:
                    conversation_history.append("\nSYSTEM: No additional relevant chunks found.\n")
                    log("  No additional relevant code chunks found")
            else:
                # LLM has provided a final answer - but check if we've met minimum iterations
                if iterations >= min_iterations:
                    log(f"\n[FINAL ANSWER PROVIDED]")
                    log(f"  After {iterations} iterations (minimum required: {min_iterations})")
                    log(f"  Response length: {len(response)} characters")
                    
                    final_answer = response
                    # Exit the loop once we have a final answer AND have met minimum iterations
                    break
                else:
                    # Force continued exploration by converting the response into a NEED_MORE_INFO request
                    logger.info(f"LLM tried to answer early at iteration {iterations}/{min_iterations}. Forcing continued exploration.")
                    log(f"\n[FORCING CONTINUED EXPLORATION]")
                    log(f"  AI tried to answer at iteration {iterations} but minimum is {min_iterations}")
                    
                    # Extract a follow-up query from the response or create a general one
                    follow_up_query = f"Find more information about {user_query}"
                    
                    log(f"  Generated follow-up query: '{follow_up_query}'")
                    
                    # Add forced exploration message to conversation history
                    conversation_history.append(f"\nLLM: {response}\n")
                    conversation_history.append(f"\nSYSTEM: You must continue exploring for at least {min_iterations} iterations. Currently at iteration {iterations}. Please find more relevant information.\n")
                    
                    # Get additional chunks with this follow-up query
                    log(f"\n[FORCING ADDITIONAL SEARCH]")
                    
                    start_time = time.time()
                    new_vector_results = collection.query.near_text(
                        query=follow_up_query,
                        limit=num_chunks,
                        return_properties=["text", "file", "category", "module"]
                    )
                    end_time = time.time()
                    
                    new_chunks = new_vector_results.objects
                    
                    log(f"  Search completed in {end_time - start_time:.2f} seconds")
                    log(f"  Found {len(new_chunks)} chunks")
                    
                    # Filter out duplicates
                    unique_new_chunks = []
                    for chunk in new_chunks:
                        chunk_id = chunk.properties.get('id', hash(chunk.properties.get('text', '')))
                        if chunk_id not in all_chunk_ids:
                            all_chunk_ids.add(chunk_id)
                            unique_new_chunks.append(chunk)
                            retrieved_files.add(chunk.properties.get('file', ''))
                    
                    log(f"  Unique new chunks: {len(unique_new_chunks)} (after filtering duplicates)")
                    
                    # Add new chunks to context
                    if unique_new_chunks:
                        additional_context = "\nADDITIONAL CHUNKS FOUND:\n\n"
                        for i, chunk in enumerate(unique_new_chunks, len(chunks)+1):
                            file_path = chunk.properties.get('file', 'unknown')
                            category = chunk.properties.get('category', 'unknown')
                            additional_context += f"CHUNK {i} (from {file_path}, category: {category}):\n```\n{chunk.properties.get('text', '')}\n```\n\n"
                        
                        # Add to conversation history
                        conversation_history.append(f"\nSYSTEM: {additional_context}\n")
                        chunks.extend(unique_new_chunks)
                        
                        log(f"  Added {len(unique_new_chunks)} new unique chunks to context")
                        log(f"  Total chunks now: {len(chunks)}")
                        log(f"  Files retrieved so far: {len(retrieved_files)}")
                    else:
                        conversation_history.append("\nSYSTEM: No additional relevant chunks found.\n")
                        log("  No additional relevant code chunks found")
            
            # Add refocusing reminder after each iteration
            if iterations > 1:
                refocus_message = f"\nSYSTEM REMINDER: The original question was '{user_query}'. Stay focused on this question.\n"
                conversation_history.append(refocus_message)
                
                log(f"\n[ADDED REFOCUSING REMINDER]")
            
            # Provide iteration summary if verbose
            log(f"\n[ITERATION {iterations} COMPLETE]")
            log(f"  Total chunks gathered: {len(chunks)}")
            log(f"  Total files retrieved: {len(retrieved_files)}")
            log(f"  Conversation history size: {len(''.join(conversation_history))} characters")
            log(f"  Moving to iteration {iterations+1} of max {max_iterations}")
            
            # At the end of each iteration
            yield {"type": "iteration_complete", "iteration": iterations, "max_iterations": max_iterations}
        
        # Final result
        yield {"type": "result", "response": final_answer if final_answer else response, "logs": logs}
        
    except Exception as e:
        logger.error(f"Error during streaming query: {e}")
        yield from log(f"\n[ERROR OCCURRED]")
        yield from log(f"  {str(e)}")
        import traceback
        yield from log(f"  Traceback: {traceback.format_exc()}")
        yield {"type": "error", "message": str(e), "logs": logs}

if __name__ == "__main__":
    # Check if Weaviate has data before starting the server
    try:
        collections = client.collections.list_all()
        has_codechunk = "CodeChunk" in collections
        
        if not has_codechunk:
            logger.error("CodeChunk class not found in Weaviate. Please run embed_codebase_weaviate.py first to index your code.")
            exit(1)
        
        collection = client.collections.get("CodeChunk")
        count = collection.aggregate.over_all().total_count
        
        if count == 0:
            logger.error("No code chunks found in Weaviate. Please run embed_codebase_weaviate.py first to index your code.")
            exit(1)
            
        logger.info(f"Found {count} code chunks in Weaviate")
        
        # Start Flask app
        logger.info("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
        
    except Exception as e:
        logger.error(f"Error checking Weaviate schema: {e}")
        logger.error("Make sure Weaviate is running and accessible at http://localhost:8080")
        exit(1)
