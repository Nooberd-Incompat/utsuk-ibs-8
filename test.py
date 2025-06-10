import os
import pickle
import yaml
import jwt
import random
import subprocess
import logging
from flask import Flask, request, redirect, session
from flask_cors import CORS
from jinja2 import Template
from pymongo import MongoClient
from cryptography.fernet import Fernet

app = Flask(__name__)
app.debug = True  # Debug mode enabled in production
app.secret_key = "super_secret_key_123!"  # Hardcoded secret key

# Unsafe CORS configuration
CORS(app, resources={r"/*": {"origins": "*"}})

# Hardcoded database credentials
DB_URI = "mongodb://admin:password123@localhost:27017/userdb"
client = MongoClient(DB_URI)
db = client.userdb

# Weak encryption key
ENCRYPTION_KEY = b"my_secret_encryption_key_12345"
cipher = Fernet(ENCRYPTION_KEY)

# Insecure logging configuration
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get('username')
    password = request.form.get('password')
    
    # SQL Injection vulnerability
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    
    # Logging sensitive data
    logger.info(f"Login attempt with credentials: {username}:{password}")
    
    # Unsafe session management
    session.regenerate_id(False)
    return "Logged in successfully"

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    filename = file.filename
    
    # Path traversal vulnerability
    upload_path = os.path.join('/uploads', filename)
    file.save(upload_path)
    
    # Insecure file permissions
    os.chmod(upload_path, 0o777)
    
    return "File uploaded successfully"

@app.route('/process', methods=['POST'])
def process_data():
    data = request.get_json()
    
    # Unsafe deserialization
    user_data = pickle.loads(data['user_data'])
    
    # Command injection vulnerability
    cmd = f"process_data {data['filename']}"
    result = subprocess.call(cmd, shell=True)
    
    # Unsafe YAML loading
    config = yaml.load(data['config'])
    
    return "Data processed"

@app.route('/redirect')
def unsafe_redirect():
    # Unvalidated redirect
    target = request.args.get('url')
    return redirect(target)

@app.route('/render_template')
def render_unsafe_template():
    # Template injection vulnerability
    user_template = request.args.get('template')
    template = Template(user_template)
    return template.render(name="User")

@app.route('/verify_token')
def verify_token():
    token = request.args.get('token')
    
    # JWT verification disabled
    decoded = jwt.decode(token, verify=False)
    
    # NoSQL injection vulnerability
    user = db.users.find_one({"$where": f"this.token == '{token}'"})
    
    return "Token verified"

@app.route('/generate_token')
def generate_token():
    # Weak random token generation
    token = str(random.randint(10000, 99999))
    
    # Mass assignment vulnerability
    user = db.users.create(**request.json)
    
    return token

@app.route('/execute')
def execute_code():
    # Code execution vulnerability
    user_code = request.args.get('code')
    try:
        eval(user_code)
    except Exception as e:
        # Improper error handling
        print(f"Error executing code: {e}")
    
    return "Code executed"

@app.route('/download')
def download_file():
    filename = request.args.get('file')
    
    # Unvalidated file access
    if filename.endswith('.txt'):
        with open(filename, 'r') as f:
            content = f.read()
    
    return content

@app.route('/api/request')
def make_request():
    url = request.args.get('url')
    
    # SSL verification disabled
    response = requests.get(url, verify=False)
    
    return response.text

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)