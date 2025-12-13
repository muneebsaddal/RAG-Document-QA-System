from flask import Flask, request, jsonify
from flask_cors import CORS
from rag_system import RAGSystem
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Initialize RAG system
# Using gemma3:270m - tiny and fast model
rag = RAGSystem(model_name="gemma3:270m")

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

ALLOWED_EXTENSIONS = {'pdf', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check if server is running"""
    return jsonify({"status": "ok", "message": "RAG server is running"})

@app.route('/api/upload', methods=['POST'])
def upload_document():
    """Upload and ingest a document"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not allowed_file(file.filename):
            return jsonify({"error": "File type not allowed. Use PDF or TXT"}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Ingest document
        rag.ingest_document(filepath)
        
        return jsonify({
            "success": True,
            "message": f"Document '{filename}' ingested successfully",
            "filename": filename
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/query', methods=['POST'])
def query_document():
    """Query the RAG system"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400
        
        question = data['question']
        
        # Get answer from RAG system
        result = rag.query(question)
        
        return jsonify({
            "success": True,
            "answer": result['answer'],
            "context": result['context']
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Reset the RAG system and clear documents"""
    try:
        global rag
        rag = RAGSystem(model_name="mistral")
        
        # Clear upload folder
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        return jsonify({
            "success": True,
            "message": "System reset successfully"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting RAG Server...")
    print("📍 Server running at: http://localhost:5000")
    print("📄 Upload documents at: http://localhost:5000/api/upload")
    print("💬 Query endpoint at: http://localhost:5000/api/query")
    app.run(debug=True, port=5000)