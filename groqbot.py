from fastapi import FastAPI, HTTPException, Request
import os
from dotenv import load_dotenv
from groq import Groq
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Retrieve Groq API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client with API key
client = Groq(api_key=GROQ_API_KEY)  # Use environment variable

@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetic Foot Thermogram Analysis API using Groq"}

@app.post("/ask")
async def ask_model(request: Request):
    """
    Route to ask the Groq Llama model questions about diabetic foot or thermogram analysis.
    """
    body = await request.json()
    user_query = body.get("query", None)

    if not user_query:
        raise HTTPException(status_code=400, detail="Query is required.")

    try:
        # Query Groq model
        completion = client.chat.completions.create(
            model="llama3-groq-70b-8192-tool-use-preview",
            messages=[
                {"role": "system", "content": "You are a highly specialized medical assistant for diabetic foot thermogram analysis. Provide medically accurate and comprehensive responses."},
                {"role": "user", "content": user_query}
            ],
            temperature=0.5,
            max_tokens=1024,
            top_p=0.85,
            stream=False  # Change to False for non-streaming response
        )

        # Collect response
        response = completion.choices[0].message.content

        return {"response": response}

    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error for debugging
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
