from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os, shutil, sqlite3, pandas as pd, uuid
import requests
import json

# Load .env secrets
load_dotenv()
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY")
DEEPINFRA_API_URL = "https://api.deepinfra.com/v1/inference/meta-llama/Llama-2-70b-chat-hf"

app = FastAPI()

# CORS for frontend on localhost:5173 (Vite default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store session-specific DB names
user_db_map = {}

class QueryRequest(BaseModel):
    query: str
    session_id: str

@app.get("/")
def read_root():
    return {"message": "Welcome to NLtoSQL API"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        session_id = str(uuid.uuid4())
        file_ext = file.filename.split(".")[-1].lower()
        file_path = f"temp/{session_id}.{file_ext}"

        # Ensure /temp folder exists
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Convert to SQLite if not already
        if file_ext in ["csv", "txt", "xls", "xlsx"]:
            df = (
                pd.read_csv(file_path)
                if file_ext in ["csv", "txt"]
                else pd.read_excel(file_path)
            )
            sqlite_path = f"temp/{session_id}.db"
            conn = sqlite3.connect(sqlite_path)
            df.to_sql("data", conn, if_exists="replace", index=False)
            conn.close()
        elif file_ext == "db":
            sqlite_path = file_path
        else:
            return {"error": "Unsupported file format"}

        user_db_map[session_id] = sqlite_path
        return {"message": "Upload successful", "session_id": session_id}

    except Exception as e:
        return {"error": f"File upload failed: {str(e)}"}

def generate_prompt(query, schema):
    """Generate a prompt for the LLM to convert natural language to SQL."""
    return f"""
You are an expert SQL developer. Convert the following natural language query to a valid SQL query based on the provided database schema.

Database Schema:
{schema}

Natural Language Query: {query}

SQL Query:
"""

def call_deepinfra_api(prompt):
    """Call the deepINFRA API to generate SQL from natural language."""
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "input": prompt,
        "temperature": 0.1,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(DEEPINFRA_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        
        # Extract the generated SQL from the response
        if "results" in result and len(result["results"]) > 0:
            generated_text = result["results"][0]["generated_text"]
            
            # Extract SQL from the generated text
            # This might need adjustment based on the actual output format
            if "SQL Query:" in generated_text:
                sql = generated_text.split("SQL Query:")[1].strip()
            else:
                sql = generated_text.strip()
                
            return {"success": True, "sql": sql}
        else:
            return {"success": False, "error": "No results returned from API"}
            
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}

@app.post("/generate-sql")
async def generate_sql(req: QueryRequest):
    try:
        db_path = user_db_map.get(req.session_id)
        if not db_path:
            return {"sql": "-- No DB found for session", "results": []}

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Extract DB schema for smarter prompting
        schema = "\n".join(
            f"{row[0]}: {row[1]}"
            for row in cursor.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table';"
            )
        )

        # Generate prompt for deepINFRA API
        prompt = generate_prompt(req.query, schema)
        
        # Call deepINFRA API
        api_response = call_deepinfra_api(prompt)
        
        if not api_response.get("success"):
            return {"sql": "-- Error calling API", "error": api_response.get("error"), "results": []}
        
        sql = api_response["sql"]
        
        # Clean up generated SQL if needed
        sql = sql.replace("```sql", "").replace("```", "").strip()

        # Execute generated SQL
        try:
            cursor.execute(sql)
            columns = [description[0] for description in cursor.description] if cursor.description else []
            raw_results = cursor.fetchall()
            
            # Convert results to dictionary format for better JSON serialization
            results = []
            for row in raw_results:
                results.append({columns[i]: value for i, value in enumerate(row)})
                
            conn.close()
            return {"sql": sql, "results": results}
        except sqlite3.Error as e:
            return {"sql": sql, "error": f"SQL execution error: {str(e)}", "results": []}
            
    except Exception as e:
        return {"sql": "-- Error generating SQL", "error": str(e), "results": []}

@app.get("/schemas/{session_id}")
async def get_schema(session_id: str):
    try:
        db_path = user_db_map.get(session_id)
        if not db_path:
            return {"success": False, "error": "No database found for this session"}
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get table information
        tables = []
        for table_info in cursor.execute("SELECT name FROM sqlite_master WHERE type='table';"):
            table_name = table_info[0]
            
            # Get column information
            columns = []
            for column_info in cursor.execute(f"PRAGMA table_info({table_name});"):
                columns.append({
                    "name": column_info[1],
                    "type": column_info[2]
                })
            
            tables.append({
                "name": table_name,
                "columns": columns
            })
        
        conn.close()
        return {"success": True, "tables": tables}
        
    except Exception as e:
        return {"success": False, "error": str(e)}