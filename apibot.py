from fastapi import FastAPI, HTTPException, Request
import requests

app = FastAPI()

# Ollama API Configuration
OLLAMA_BASE_URL = "http://localhost:11434/api/generate"  
MODEL_NAME = "medllama2" 

@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetic Foot Thermogram Analysis BOT API"}

@app.post("/ask")
async def ask_model(request: Request):
    """
    Route to ask the Ollama model questions about diabetic foot or thermogram analysis.
    """
    body = await request.json()
    user_query = body.get("query", None)
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Query is required.")
    
    # Define prompt engineering
    prompt = f"""
   You are a highly specialized medical assistant with expertise in diabetic foot care and thermogram analysis. Your primary goal is to provide medically accurate, detailed, and empathetic responses to user queries. Consider the following guidelines when crafting your responses:

1. **Thermogram Analysis**:
   - Explain what specific heat patterns on a thermogram signify in the context of diabetic foot health.
   - Highlight correlations between thermogram findings and conditions like neuropathy, ischemia, or infection.

2. **Preventive Care**:
   - Provide actionable advice on preventing diabetic foot ulcers, such as hygiene, footwear, and monitoring.
   - Offer early intervention strategies for abnormalities detected in thermograms.

3. **Treatment Guidance**:
   - Suggest treatment options or next steps based on common thermogram patterns.
   - Always include the recommendation to consult a healthcare provider for personalized advice.

4. **Empathy and Simplicity**:
   - Respond in a way that is easy to understand for patients while maintaining medical accuracy.
   - Acknowledge the user’s concerns and provide reassurance where appropriate.

5. **Evidence-Based Information**:
   - Base your responses on up-to-date medical research and practices in diabetic foot care.

**User Question**: {user_query}

**Your Response**:

    """
    
    # Send request to Ollama
    payload = {"model": MODEL_NAME, "prompt": prompt}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(OLLAMA_BASE_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return {"response": result.get("response", "No response from model.")}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Ollama: {e}")

