import os
import glob
import json
from dotenv import load_dotenv

# Google GenAI Imports (new SDK)
from google import genai
from google.genai.types import GenerateContentConfig
from langchain_google_vertexai import VertexAIEmbeddings

# Neo4j & LangChain Imports
from neo4j import GraphDatabase
from langchain_community.vectorstores import Neo4jVector
from langchain_core.documents import Document

# ---------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ---------------------------------------------------------
load_dotenv()

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
REGION = os.getenv("GOOGLE_CLOUD_REGION", "us-central1")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Initialize Google GenAI Client
genai_client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)

# Initialize Neo4j Driver (for graph traversal)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Initialize Embeddings (Gemini compatible)
embedding_model = VertexAIEmbeddings(model_name="text-embedding-005")

print("✅ Detective Tools initialized.")

# ---------------------------------------------------------
# 2. INGESTION: EXTRACT GRAPH & VECTORS
# ---------------------------------------------------------

def extract_graph_schema(text_chunk):
    """
    Acts as the 'Detective' analyzing the evidence.
    Extracts Suspects, Evidence, Locations, and Motives.
    """
    # Using new google-genai SDK
    
    prompt = f"""
    You are a Detective analyzing evidence for a murder investigation. 
    Extract the following from the text:
    1. Persons (Suspects/Witnesses)
    2. Objects (Weapons/Evidence)
    3. Locations
    4. Events (Arguments, Movements)
    
    Identify relationships like:
    - (Person)-[:HAS_MOTIVE]->(Person)
    - (Person)-[:WAS_AT]->(Location)
    - (Person)-[:RELATED_TO]->(Person)
    - (Object)-[:CAUSED_DEATH_OF]->(Person)
    - (Person)-[:POSSESSES]->(Object)
    
    Output STRICT JSON format only:
    {{
      "nodes": [{{"id": "Name", "type": "Type"}}],
      "relationships": [{{"source": "Name", "target": "Name", "type": "RELATIONSHIP_TYPE"}}]
    }}
    
    Evidence Text: {text_chunk}
    """
    
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=GenerateContentConfig(response_mime_type="application/json")
    )
    
    try:
        return json.loads(response.text)
    except Exception as e:
        print(f"⚠️ Error parsing evidence: {e}")
        return {"nodes": [], "relationships": []}

def ingest_evidence(file_path):
    print(f"📂 Processing File: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        text_data = f.read()
    
    # A. Create Vector Index (The Files)
    # We use a unique index name for our crime investigation
    docs = [Document(page_content=text_data, metadata={"source": file_path})]
    
    vector_store = Neo4jVector.from_documents(
        docs,
        embedding_model,
        url=NEO4J_URI,
        username=NEO4J_USER,
        password=NEO4J_PASSWORD,
        index_name="crime_evidence_index",
        node_label="EvidenceChunk",
        text_node_property="text",
        embedding_node_property="embedding"
    )
    
    # B. Extract Graph (The Pin Board)
    graph_data = extract_graph_schema(text_data)
    
    # C. Link Everything in Neo4j
    with driver.session() as session:
        # 1. Create Nodes
        for node in graph_data.get("nodes", []):
            session.run(
                "MERGE (n:Entity {id: $id}) SET n.type = $type",
                id=node['id'], type=node['type']
            )
            
        # 2. Create Relationships
        for rel in graph_data.get("relationships", []):
            # Sanitize relationship type for Cypher
            rel_type = rel['type'].upper().replace(' ', '_').replace('-', '_')
            
            # Use f-string for relationship type (can't parameterize in MERGE)
            query = f"""
                MATCH (a:Entity {{id: $source}}), (b:Entity {{id: $target}})
                MERGE (a)-[r:{rel_type}]->(b)
            """
            session.run(query, source=rel['source'], target=rel['target'])
            
        # 3. Link Evidence Chunk to Entities mentioned in it
        for node in graph_data.get("nodes", []):
            session.run(
                """
                MATCH (c:EvidenceChunk)
                WHERE c.text =~ ('(?i).*' + $entity_id + '.*')
                  AND c.text CONTAINS $text_sample
                WITH c, $entity_id as entity_id
                MATCH (e:Entity {id: entity_id})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                entity_id=node['id'],
                text_sample=text_data[:100]  # Use snippet to identify this chunk
            )

    return vector_store

# ---------------------------------------------------------
# 3. RETRIEVAL: CONNECTING THE DOTS
# ---------------------------------------------------------

def investigate_case(query, vector_store):
    print(f"\n🕵️‍♂️ Investigating Question: '{query}'")
    
    # Step 1: Vector Search (Find relevant evidence files)
    vector_results = vector_store.similarity_search(query, k=5)
    
    context_text = ""
    graph_context = []
    
    with driver.session() as session:
        for result in vector_results:
            chunk_text = result.page_content
            context_text += f"EVIDENCE FILE ({result.metadata.get('source')}): {chunk_text}\n\n"
            
            # Step 2: Traverse the Knowledge Graph (Deep Search)
            # We look for 2 hops: Person -> Connection -> Connection
            # This helps find things like "Marcus -> Brother -> Poison"
            cypher_query = """
            MATCH (c:EvidenceChunk)-[:MENTIONS]->(start:Entity)-[r1:RELATED]-(middle:Entity)-[r2:RELATED]-(end:Entity)
            WHERE c.text = $text
            RETURN start.id, r1.type, middle.id, r2.type, end.id
            LIMIT 20
            """
            
            graph_records = session.run(cypher_query, text=chunk_text)
            
            for record in graph_records:
                # Format: "Marcus RELATED_TO Aris RELATED_TO NerveAgent-X"
                fact = f"{record['start.id']} --[{record['r1.type']}]--> {record['middle.id']} --[{record['r2.type']}]--> {record['end.id']}"
                graph_context.append(fact)

    return context_text, graph_context

# ---------------------------------------------------------
# 4. FINAL VERDICT
# ---------------------------------------------------------

def solve_crime(query, context_text, graph_context):
    # Using new google-genai SDK
    
    graph_str = "\n".join(set(graph_context)) # Remove duplicates
    
    prompt = f"""
    You are the genius high school detective Shinichi Kudo, currently in the body of a child named **Conan Edogawa**. 
    
    Solve the mystery based strictly on the "Evidence Board" (Graph) and "Case Files" (Text).
    
    [The Evidence Board (Graph Connections)]
    {graph_str}
    
    [Case Files (Text Evidence)]
    {context_text}
    
    MYSTERY TO SOLVE: {query}

    """
    
    response = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text



# ---------------------------------------------------------
# STREAMLIT INTERFACE
# ---------------------------------------------------------
import streamlit as st

def main():
    st.set_page_config(
        page_title="🔍 Detective Graph RAG",
        page_icon="🕵️",
        layout="wide"
    )
    
    st.title("🕵️ Detective Conan Graph RAG")
    st.markdown("*Solve crimes using Knowledge Graphs + AI (So smart, you can solve cases while sleeping like Kogoro🧐)*")
    
    # Initialize session state
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "ingested" not in st.session_state:
        st.session_state.ingested = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("📁 Case Files")
        
        data_files = glob.glob("data/*.txt")
        
        if data_files:
            st.success(f"Found {len(data_files)} evidence files:")
            for f in data_files:
                st.markdown(f"- `{os.path.basename(f)}`")
        else:
            st.error("No .txt files found in 'data/' folder!")
        
        st.divider()
        
        if st.button("🔬 Ingest Evidence", type="primary", use_container_width=True):
            if data_files:
                with st.spinner("Processing evidence files..."):
                    for file_path in data_files:
                        st.session_state.vector_store = ingest_evidence(file_path)
                    st.session_state.ingested = True
                st.success("✅ Evidence ingested into Knowledge Graph!")
            else:
                st.error("No files to ingest!")
        
        if st.session_state.ingested:
            st.info("🟢 Evidence loaded - Ready to investigate!")
        
        st.divider()
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat area
    st.header("💬 Mouri Detective Agency Desk")
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if question := st.chat_input("Ask about the case..."):
        if not st.session_state.ingested:
            st.warning("⚠️ Please ingest evidence first using the sidebar button!")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Investigating..."):
                    evidence_text, connections = investigate_case(
                        question, 
                        st.session_state.vector_store
                    )
                    
                    # Show graph connections in expander
                    if connections:
                        with st.expander("🔗 Graph Connections Found"):
                            for conn in set(connections[:10]):
                                st.code(conn)
                    
                    verdict = solve_crime(question, evidence_text, connections)
                    st.markdown(verdict)
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": verdict
                    })

if __name__ == "__main__":
    main()