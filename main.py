from flask import Flask, request, jsonify
import requests, os, json, re, chromadb
from dotenv import load_dotenv
import google.generativeai as genai
from chromadb.utils import embedding_functions
from datetime import datetime
import json as pyjson

load_dotenv()

app = Flask(__name__)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

API_KEY = os.getenv("api_key")
genai.configure(api_key=API_KEY)

# Loading embedding model
genai_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(
    api_key=API_KEY, model_name='text-embedding-004'
)

# Initializing chromadb
client = chromadb.Client()
collection = client.create_collection("mental_wellness")

with open('resources.json', 'r') as f:
    dataset = json.load(f)

NORMALIZED = {k.lower().strip(): v for k, v in dataset.items()}

# ---------------- In-memory Mood Log ---------------- #
mood_log = []

# ---------------- Utility ---------------- #
def merge_lines(lines):
    cleaned = [re.sub(r"\s+", " ", s).strip() for s in lines if s and s.strip()]
    return " ".join(cleaned)

# ---------------- Knowledge Base ---------------- #
def build_knowledge_base():
    documents, metadatas, ids = [], [], []

    for keyword, responses in dataset.items():
        merged_text = " ".join(responses)
        documents.append(merged_text)
        metadatas.append({'keyword': keyword, 'type': "curated_response"})
        ids.append(f"keyword_{keyword.replace(' ', '_')}")

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f'Added {len(documents)} documents to knowledge base')

build_knowledge_base()


def retrieve_relevant_context(query, n_results=3):
    results = collection.query(query_texts=[query], n_results=n_results)

    context_pieces = []
    for i, doc in enumerate(results['documents'][0]):
        keyword = results['metadatas'][0][i]['keyword']
        distance = results["distances"][0][i]
        if distance < 1.5:
            context_pieces.append({
                "keyword": keyword,
                "content": doc,
                "relevance_score": 1 - distance
            })
    return context_pieces


def dynamic_context_selection(context_pieces, min_relevance=-0.4, max_gap=0.2):
    if not context_pieces:
        return []

    sorted_pieces = sorted(context_pieces, key=lambda x: x['relevance_score'], reverse=True)
    selected = []

    for i, piece in enumerate(sorted_pieces):
        if piece['relevance_score'] < min_relevance:
            break
        if i > 0:
            gap = sorted_pieces[i-1]['relevance_score'] - piece['relevance_score']
            if gap > max_gap:
                break
        selected.append(piece)
        if len(selected) >= 4:
            break

    return selected


def create_enhanced_prompt(user_message, context_pieces):
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


# ---------------- Gemini Call Helpers ---------------- #
def get_gemini_reply(prompt):
    headers = {"Content-Type": "application/json"}
    params = {"key": API_KEY}
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(GEMINI_URL, params=params, json=body, headers=headers, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print(f"Error fetching Gemini response: {e}")
        return "Sorry, I couldn't generate a response right now."


# ---------------- Mood Detection ---------------- #
def detect_mood(text):
    mood_prompt = f"""
    Analyze the user's message and classify their mood into one of:
    [happy, sad, anxious, angry, neutral, stressed, excited, tired].

    USER MESSAGE: "{text}"

    Respond ONLY in JSON:
    {{
      "mood": "<one of the above>",
      "confidence": <number between 0 and 1>
    }}
    """
    try:
        response = model.generate_content([{"text": mood_prompt}])
        result = response.text.strip()

        match = re.search(r'\{.*\}', result, re.DOTALL)
        if match:
            result = match.group(0)

        mood_data = pyjson.loads(result)
        mood = mood_data.get("mood", "neutral").lower().strip()
        confidence = float(mood_data.get("confidence", 0.5))

        valid_moods = ["happy", "sad", "anxious", "angry", "neutral", "stressed", "excited", "tired"]
        if mood not in valid_moods:
            mood = "neutral"

        return mood, confidence

    except Exception as e:
        print("Mood detection failed:", e)
        return "neutral", 0.5


def log_mood(mood):
    entry = {"mood": mood, "timestamp": datetime.now().isoformat()}
    mood_log.append(entry)
    return entry


@app.post("/mood/confirm")
def confirm_mood():
    payload = request.get_json(force=True)
    mood = payload.get("mood", "").lower()
    correct = payload.get("correct", False)

    if correct and mood:
        entry = log_mood(mood)
        return jsonify({"message": f"Mood '{mood}' confirmed and logged.", "entry": entry})
    else:
        return jsonify({"message": "Okay, please tell me how you are feeling.", "needs_user_input": True})


# ---------------- Chat Endpoint ---------------- #
@app.post("/chat")
def chat():
    payload = request.get_json(force=True)
    user_msg = payload.get("message", "").strip()

    if not user_msg:
        return jsonify({"error": "message is required"}), 400

    # Step 1: Detect mood
    detected_mood, confidence = detect_mood(user_msg)
    latest_mood = None

    if confidence >= 0.75:
        entry = log_mood(detected_mood)
        latest_mood = detected_mood
    else:
        return jsonify({
            "response": f"I sense you might be feeling {detected_mood}. Is that correct?",
            "mood_detected": detected_mood,
            "confidence": confidence,
            "needs_confirmation": True
        })

    # Step 2: Retrieve context
    context_pieces = retrieve_relevant_context(user_msg, n_results=5)
    selected_contexts = dynamic_context_selection(context_pieces)

    try:
        if selected_contexts:
            enhanced_prompt = create_enhanced_prompt(
                user_msg + f"\n\nThe user’s latest mood is '{latest_mood}'.",
                selected_contexts
            )
            response = model.generate_content([enhanced_prompt])
            ai_reply = response.text
            return jsonify({
                "response": ai_reply,
                "mood_logged": latest_mood,
                "confidence": confidence,
                "method": "dynamic_synthesis",
                "contexts_found": len(context_pieces),
                "contexts_used": len(selected_contexts),
                "keywords_used": [ctx['keyword'] for ctx in selected_contexts]
            })
        else:
            response = model.generate_content([user_msg])
            return jsonify({
                "response": response.text,
                "mood_logged": latest_mood,
                "confidence": confidence,
                "method": "llm_fallback",
                "contexts_found": len(context_pieces)
            })
    except Exception as e:
        response = get_gemini_reply(user_msg)
        return jsonify({
            "response": response,
            "mood_logged": latest_mood,
            "confidence": confidence,
            "method": "simple_fallback"
        })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)