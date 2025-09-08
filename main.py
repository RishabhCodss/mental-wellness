from flask import Flask, request, jsonify
import requests, os, json, re, chromadb
from dotenv import load_dotenv
import google.generativeai as genai
from chromadb.utils import embedding_functions
load_dotenv()


app = Flask(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     user_message = data['message']
#     # Call Gemini API here
#     ai_response = get_gemini_reply(user_message)  # you implement this
#     return jsonify({'response': ai_response})

# if __name__ == '__main__':
#     app.run(debug=True)

API_KEY = os.getenv("api_key")

genai.configure(api_key=API_KEY)

#Loading embedding model
genai_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key= API_KEY, model_name='text-embedding-004'
)

# Initializing chromadb
client = chromadb.Client()
collection = client.create_collection("mental_wellness")

with open('resources.json', 'r') as f:
        dataset = json.load(f)

NORMALIZED = {k.lower().strip(): v for k,v in dataset.items()}

def merge_lines(lines):
    """Convert multiple response lines into single merged response"""
    cleaned = [re.sub(r"\s+", " ", s).strip() for s in lines if s and s.strip()]
    return " ".join(cleaned)

def find_keyword_responses(text):
    """
    Look for exact keyword matches in user message
    Returns curated response if found, None otherwise
    """
    user_text = text.lower()
    
    # Try exact phrase match first
    if user_text in NORMALIZED:
        return merge_lines(NORMALIZED[user_text])
    
    # Try substring/contains match
    for keyword, response_lines in NORMALIZED.items():
        if keyword in user_text:
            return merge_lines(response_lines)
    
    return None  # No match found

def build_knowledge_base():
    
    documents = []
    metadatas = []
    ids = []

    for keyword, responses in dataset.items():
        merged_text = " ".join(responses)

        documents.append(merged_text)
        metadatas.append({'keyword' : keyword , 'type' : "curated_response"})
        ids.append(f"keyword_{keyword.replace(' ', '_')}")
    
    #Added documents to chromadb
    collection.add(
        documents = documents,
        metadatas = metadatas,
        ids = ids
    ) 
    print(f'Added {len(documents)} documents to knowledge base')

build_knowledge_base()

def retrieve_relevant_context(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    print(f"DEBUG: Query: '{query}'")
    print(f"DEBUG: Raw distances: {results['distances'][0]}")
    print(f"DEBUG: Keywords found: {[results['metadatas'][0][i]['keyword'] for i in range(len(results['documents'][0]))]}")
    
    context_pieces = []
    for i, doc in enumerate(results['documents'][0]):
        keyword = results['metadatas'][0][i]['keyword']
        distance = results["distances"][0][i]
        print(f"DEBUG: Keyword '{keyword}' has distance {distance}")
        
        if distance < 1.5:  # Your current threshold
            context_pieces.append({
                "keyword": keyword,
                "content": doc,
                "relevance_score": 1 - distance
            })
    
    print(f"DEBUG: Found {len(context_pieces)} matches under threshold 1.5")
    return context_pieces

def dynamic_context_selection(context_pieces, min_relevance=-0.4, max_gap=0.2):
    
    if not context_pieces:
        return []
    
    # Sort by relevance score (highest first)
    sorted_pieces = sorted(context_pieces, key=lambda x: x['relevance_score'], reverse=True)
    selected = []
    
    for i, piece in enumerate(sorted_pieces):
        # Skip if below minimum relevance threshold
        if piece['relevance_score'] < min_relevance:
            break
            
        # Check for large gap between consecutive pieces (indicates drop in relevance)
        if i > 0:
            gap = sorted_pieces[i-1]['relevance_score'] - piece['relevance_score']
            if gap > max_gap:
                break
                
        selected.append(piece)
        
        # Cap at 4 pieces to prevent overwhelming responses
        if len(selected) >= 4:
            break
    
    return selected

def create_enhanced_prompt(user_message, context_pieces):
    """Enhanced prompt that asks LLM to synthesize multiple contexts"""
    system_prompt = """You are a compassionate youth mental wellness assistant.
    
    The user has asked about mental health concerns. I've found several relevant topics from our knowledge base.
    
    Your task:
    1. Synthesize the provided information into a coherent, personalized response
    2. Address the connections between multiple mental health topics if relevant
    3. Provide actionable, supportive guidance
    4. Keep the response warm and encouraging
    
    Important: Don't just list separate advice for each topic - weave them together thoughtfully."""

    if context_pieces:
        context_text = "\n\n".join([
            f"**{piece['keyword'].upper()}** (relevance: {piece['relevance_score']:.2f}):\n{piece['content']}"
            for piece in context_pieces
        ])
        
        enhanced_prompt = f"""{system_prompt}

RELEVANT KNOWLEDGE BASE CONTENT:
{context_text}

USER QUESTION: {user_message}

SYNTHESIZED RESPONSE:"""
    else:
        enhanced_prompt = f"""{system_prompt}

USER QUESTION: {user_message}

RESPONSE:"""
    
    return enhanced_prompt

model = genai.GenerativeModel('gemini-2.5-flash')


def get_gemini_reply(prompt):
    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}
    body = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    try:
        r = requests.post(GEMINI_URL, params=params, json=body, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Error fetching Gemini response: {e}")
        return "Sorry, I couldn't generate a response right now."



@app.post("/chat")
def chat():
    payload = request.get_json(force=True)
    user_msg = payload.get("message", "").strip()
    
    if not user_msg:
        return jsonify({"error": "message is required"}), 400

    try:
        # Use embedding-based context retrieval for all queries
        context_pieces = retrieve_relevant_context(user_msg, n_results=5)
        
        # Dynamic selection of relevant contexts
        selected_contexts = dynamic_context_selection(context_pieces)
        
        if selected_contexts:
            # Create enhanced prompt for LLM synthesis
            enhanced_prompt = create_enhanced_prompt(user_msg, selected_contexts)
            
            # Generate response with context synthesis
            response = model.generate_content([enhanced_prompt])
            ai_reply = response.text
            
            return jsonify({
                "response": ai_reply,
                "method": "dynamic_synthesis",
                "contexts_found": len(context_pieces),
                "contexts_used": len(selected_contexts),
                "keywords_used": [ctx['keyword'] for ctx in selected_contexts]
            })
        else:
            # Fallback to general LLM response
            response = model.generate_content([user_msg])
            return jsonify({
                "response": response.text,
                "method": "llm_fallback",
                "contexts_found": len(context_pieces)
            })
            
    except Exception as e:
        # Final fallback
        response = get_gemini_reply(user_msg)
        return jsonify({
            "response": response,
            "method": "simple_fallback"
        })
    

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
