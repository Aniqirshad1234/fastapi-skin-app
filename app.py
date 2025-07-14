# import os
# import sqlite3
# import numpy as np
# from flask import Flask, request, render_template, redirect, url_for, session, flash
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from werkzeug.utils import secure_filename
# from werkzeug.security import generate_password_hash, check_password_hash

# # App setup
# app = Flask(__name__)
# app.secret_key = os.urandom(24)  # Secure random secret key
# app.config['UPLOAD_FOLDER'] = 'static/uploads'
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB upload limit
# app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# # Load model and class names
# model = load_model("best_skin_cancer_model.h5")
# class_names = ['Benign', 'Malignant']

# # Database setup
# def init_db():
#     conn = sqlite3.connect('users.db')
#     c = conn.cursor()
#     c.execute('''CREATE TABLE IF NOT EXISTS users 
#                 (id INTEGER PRIMARY KEY, 
#                 username TEXT UNIQUE, 
#                 password TEXT,
#                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
#     conn.commit()
#     conn.close()

# # Check allowed file extensions
# def allowed_file(filename):
#     return '.' in filename and \
#            filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# # Image preprocessing
# def preprocess_image(image_path):
#     img = load_img(image_path, target_size=(224, 224))
#     img_array = img_to_array(img)
#     img_array = img_array / 255.0
#     return np.expand_dims(img_array, axis=0)

# # Home page (login required)
# @app.route('/')
# def home():
#     if 'username' in session:
#         return render_template('index.html', username=session['username'])
#     return redirect(url_for('login'))

# # Signup
# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if 'username' in session:
#         return redirect(url_for('home'))
    
#     if request.method == 'POST':
#         username = request.form.get('username', '').strip()
#         password = request.form.get('password', '')
#         confirm_password = request.form.get('confirm_password', '')
        
#         # Validate inputs
#         if not username or not password or not confirm_password:
#             flash('All fields are required!', 'error')
#             return redirect(url_for('signup'))
        
#         if len(username) < 4:
#             flash('Username must be at least 4 characters long', 'error')
#             return redirect(url_for('signup'))
            
#         if len(password) < 8:
#             flash('Password must be at least 8 characters long', 'error')
#             return redirect(url_for('signup'))
            
#         if password != confirm_password:
#             flash('Passwords do not match!', 'error')
#             return redirect(url_for('signup'))
        
#         # Check if username exists
#         conn = sqlite3.connect('users.db')
#         c = conn.cursor()
#         c.execute("SELECT id FROM users WHERE username=?", (username,))
#         if c.fetchone():
#             conn.close()
#             flash('Username already exists!', 'error')
#             return redirect(url_for('signup'))
        
#         # Create new user
#         hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
#         try:
#             c.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
#                      (username, hashed_password))
#             conn.commit()
#             flash('Account created successfully! Please log in.', 'success')
#             return redirect(url_for('login'))
#         except Exception as e:
#             conn.rollback()
#             flash('An error occurred during registration', 'error')
#             return redirect(url_for('signup'))
#         finally:
#             conn.close()
    
#     return render_template('signup.html')

# # Login
# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if 'username' in session:
#         return redirect(url_for('home'))
    
#     if request.method == 'POST':
#         username = request.form.get('username', '').strip()
#         password = request.form.get('password', '')
        
#         if not username or not password:
#             flash('Please enter both username and password', 'error')
#             return redirect(url_for('login'))
        
#         conn = sqlite3.connect('users.db')
#         c = conn.cursor()
#         c.execute("SELECT id, username, password FROM users WHERE username=?", (username,))
#         user = c.fetchone()
#         conn.close()
        
#         if user and check_password_hash(user[2], password):
#             session['username'] = user[1]
#             session['user_id'] = user[0]
#             flash('Logged in successfully!', 'success')
#             next_page = request.args.get('next') or url_for('home')
#             return redirect(next_page)
#         else:
#             flash('Invalid username or password', 'error')
    
#     return render_template('login.html')

# # Logout
# @app.route('/logout')
# def logout():
#     session.clear()
#     flash('You have been logged out.', 'info')
#     return redirect(url_for('login'))

# # Predict Route
# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'username' not in session:
#         return redirect(url_for('login'))
    
#     if 'file' not in request.files:
#         flash('No file selected', 'error')
#         return redirect(url_for('home'))
    
#     file = request.files['file']
#     if file.filename == '':
#         flash('No file selected', 'error')
#         return redirect(url_for('home'))
    
#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         if not os.path.exists(app.config['UPLOAD_FOLDER']):
#             os.makedirs(app.config['UPLOAD_FOLDER'])
        
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         file.save(file_path)
        
#         try:
#             img = preprocess_image(file_path)
#             prediction = model.predict(img)
#             predicted_class = class_names[np.argmax(prediction)]
#             confidence = float(np.max(prediction)) * 100
            
#             return render_template('index.html', 
#                                  prediction=predicted_class,
#                                  confidence=f"{confidence:.2f}%",
#                                  image_path=file_path,
#                                  username=session['username'])
#         except Exception as e:
#             flash('Error processing image. Please try another.', 'error')
#             return redirect(url_for('home'))
    
#     flash('Allowed file types are png, jpg, jpeg', 'error')
#     return redirect(url_for('home'))

# if __name__ == "__main__":
#     init_db()
#     app.run(debug=True)


import os
import uvicorn
import numpy as np
from fastapi import FastAPI, Request, Form, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from starlette.middleware.sessions import SessionMiddleware
import shutil
import sqlite3
from passlib.hash import pbkdf2_sha256

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.urandom(24))

# Set static and template directories
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Configs
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Load the trained model
model = load_model("best_skin_cancer_model.h5")
class_names = ["Benign", "Malignant"]

# Utility: Check file extension
def allowed_file(filename: str):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Utility: Preprocess image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Utility: Database init
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users 
                 (id INTEGER PRIMARY KEY, 
                 username TEXT UNIQUE, 
                 password TEXT,
                 created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

# Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    username = request.session.get("username")
    prediction = request.session.get("prediction")
    confidence = request.session.get("confidence")
    image_path = request.session.get("image_path")

    if username:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "username": username,
            "prediction": prediction,
            "confidence": confidence,
            "image_path": image_path
        })
    return RedirectResponse(url="/login", status_code=302)

# Signup
@app.get("/signup", response_class=HTMLResponse)
def signup_form(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.post("/signup")
def signup(request: Request, username: str = Form(...), password: str = Form(...), confirm_password: str = Form(...)):
    if password != confirm_password:
        return RedirectResponse("/signup", status_code=302)

    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT id FROM users WHERE username=?", (username,))
    if c.fetchone():
        conn.close()
        return RedirectResponse("/signup", status_code=302)

    hashed_password = pbkdf2_sha256.hash(password)

    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()
    return RedirectResponse("/login", status_code=302)

# Login
@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT id, password FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    if user and pbkdf2_sha256.verify(password, user[1]):
        request.session["username"] = username
        return RedirectResponse("/", status_code=302)
    return RedirectResponse("/login", status_code=302)

# Logout
@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)

# Predict
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    username = request.session.get("username")
    if not username:
        return RedirectResponse("/login", status_code=302)

    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file format.")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        img = preprocess_image(file_path)
        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        request.session["prediction"] = predicted_class
        request.session["confidence"] = f"{confidence:.2f}"
        request.session["image_path"] = "/" + file_path.replace("\\", "/")

        return RedirectResponse("/", status_code=302)

    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed.")

# Start DB
init_db()
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

