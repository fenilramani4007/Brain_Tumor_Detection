from flask import Flask, render_template, request, redirect, url_for, flash
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import os
from werkzeug.utils import secure_filename
from math import ceil
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField
from wtforms.validators import InputRequired
from bson.objectid import ObjectId
import cv2
import numpy as np
from keras.models import load_model

# App Initialization
app = Flask(__name__)
app.config['SECRET_KEY'] = 'abc1234'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/brain_tumor_db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

mongo = PyMongo(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load Model
model = load_model('brain_tumor_model.h5')
labels = ["glioma", "no_tumor", "meningioma", "pituitary"]

# User Model
class User(UserMixin):
    def __init__(self, user_id, username, role):
        self.id = str(user_id)
        self.username = username
        self.role = role

    def is_admin(self):
        return self.role == 'admin'

@login_manager.user_loader
def load_user(user_id):
    user = mongo.db.users.find_one({"_id": ObjectId(user_id)})
    return User(user['_id'], user['username'], user['role']) if user else None

# Utility Functions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Resize the image to (224, 224) as required by MobileNetV2
    img = cv2.resize(img, (224, 224))
    
    # Normalize the image
    img = img / 255.0
    
    # Expand dimensions to fit the model input (batch size of 1)
    img = np.expand_dims(img, axis=0)
    return img

# Routes
@app.route('/')
def index():
    return render_template('index.html')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        role = request.form['role']  # Keeping role as is since it's from a select input
        
        if mongo.db.users.find_one({"username": username}):
            flash('Username already exists!', 'danger')
        else:
            hashed_password = generate_password_hash(password)
            mongo.db.users.insert_one({"username": username, "password": hashed_password, "role": role})
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))
    
    return render_template('register.html', form=form)


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired()])
    password = PasswordField('Password', validators=[InputRequired()])

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        username = form.username.data
        password = form.password.data
        user = mongo.db.users.find_one({"username": username})

        if user and check_password_hash(user['password'], password):
            login_user(User(user['_id'], user['username'], user.get('role', 'user')))

            # Redirect Admins to the Admin Dashboard
            if user.get('role') == 'admin':
                return redirect(url_for('admin_dashboard'))
            
            return redirect(url_for('dashboard'))
        
        flash('Invalid credentials!', 'danger')

    return render_template('login.html', form=form)


@app.route('/dashboard')
@login_required
def dashboard():
    # Fetch user activities (tumor prediction history) from the database
    activities = mongo.db.activities.find({"username": current_user.username})  # Assuming activities are linked by username
    
    # Fetch the user's appointments
    appointments = mongo.db.appointments.find({"patient_name": current_user.username})
    
    # Generate an AI suggestion for the user
    ai_suggestion = "Based on your previous tumor predictions, it is recommended to monitor your health regularly and follow up with your healthcare provider."
    
    # Paginate the results
    page = request.args.get('page', 1, type=int)
    per_page = 8
    total_activities = mongo.db.activities.count_documents({"username": current_user.username})  # Use count_documents instead of count
    total_pages = (total_activities + per_page - 1) // per_page
    activities_paginated = activities.skip((page - 1) * per_page).limit(per_page)

    return render_template('user_dashboard.html', 
                           activities=activities_paginated, 
                           appointments=appointments, 
                           ai_suggestion=ai_suggestion,
                           page=page, 
                           total_pages=total_pages)

# Fetch User Suggestions (AJAX)
@app.route('/search_user_suggestions', methods=['GET'])
@login_required
def search_user_suggestions():
    if current_user.role != 'admin':
        return jsonify([])

    search_term = request.args.get('query', '').strip()
    if not search_term:
        return jsonify([])

    users = mongo.db.users.find({"username": {"$regex": search_term, "$options": "i"}}, {"username": 1}).limit(5)
    user_list = [user["username"] for user in users]

    return jsonify(user_list)


# Admin Dashboard Route
@app.route('/admin_dashboard', methods=['GET', 'POST'])
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('index'))

    # Pagination settings
    page = int(request.args.get('page', 1))
    per_page = 5
    skip = (page - 1) * per_page

    # Fetch pending appointments with pagination
    total_appointments = mongo.db.appointments.count_documents({"status": "Pending"})
    appointments = list(mongo.db.appointments.find({"status": "Pending"}).skip(skip).limit(per_page))

    # Fetch user search query
    search_query = request.args.get('search', '')

    user_data = []
    if search_query:
        user = mongo.db.users.find_one({"username": search_query})
        if user:
            user_data = list(mongo.db.appointments.find({"patient_name": search_query}))

    total_pages = (total_appointments + per_page - 1) // per_page

    return render_template('admin_dashboard.html', 
                           appointments=appointments, 
                           user_data=user_data, 
                           search_query=search_query, 
                           page=page, 
                           total_pages=total_pages)

@app.route('/nearest_hospitals')
@login_required
def nearest_hospitals():
    # Example: This is just a mock list of hospitals for demonstration
    hospitals = [
        {"name": "City Hospital", "address": "123 City St.", "contact": "123-456-7890"},
        {"name": "Central Medical Center", "address": "456 Central Ave.", "contact": "987-654-3210"},
        {"name": "HealthCare Clinic", "address": "789 Health Rd.", "contact": "555-123-4567"}
    ]
    
    return render_template('nearest_hospitals.html', hospitals=hospitals)


from datetime import datetime

@app.route('/predict', methods=['GET', 'POST'])
@login_required
def predict():
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(filepath)
            img = preprocess_image(filepath)
            prediction = model.predict(img)
            class_index = int(np.argmax(prediction))
            class_label = labels[class_index]
            confidence = float(prediction[0][class_index] * 100)

            # Save the prediction result to MongoDB
            mongo.db.activities.insert_one({
                "username": current_user.username,
                "prediction": class_label,
                "confidence": confidence,
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image": file.filename  # Save the image filename in the database
            })
            return render_template('result.html', label=class_label, confidence=confidence, filepath=filepath)
        flash("Invalid file type!", "danger")
    return render_template('predict.html')



@app.route('/appointment', methods=['GET', 'POST'])
@login_required
def appointment():
    if request.method == 'POST':
        mongo.db.appointments.insert_one({
            "doctor_name": request.form['doctor_name'],
            "date": request.form['date'],
            "time": request.form['time'],
            "patient_name": current_user.username,
            "question": request.form['question'],
            "status": "Pending"
        })
        flash('Appointment booked successfully!', 'success')
        return redirect(url_for('dashboard'))
    return render_template('appointment.html')


# Approve Appointment
@app.route('/admin/approve/<appointment_id>', methods=['POST'])
@login_required
def approve_appointment(appointment_id):
    if current_user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('dashboard'))

    medicine = request.form['medicine']

    mongo.db.appointments.update_one(
        {"_id": ObjectId(appointment_id)},
        {"$set": {"status": "Approved", "medicine": medicine}}
    )

    flash('Appointment approved successfully!', 'success')
    return redirect(url_for('admin_dashboard'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'success')
    return redirect(url_for('login'))


@app.context_processor
def inject_navigation():
    if current_user.is_authenticated:
        nav_links = [
            {"title": "Home", "url": url_for('index')},
            {"title": "Predict Tumor", "url": url_for('predict')},
            {"title": "Book Appointment", "url": url_for('appointment')},
            {"title": "Nearest Hospitals", "url": url_for('nearest_hospitals')},
            {"title": "Dashboard", "url": url_for('dashboard')},  
            {"title": "Logout", "url": url_for('logout')}
        ]
        if current_user.role == 'admin':
            nav_links.append({"title": "Admin Dashboard", "url": url_for('admin_dashboard')})

        return dict(nav_links=nav_links)

    else:
        nav_links = [
            {"title": "Home", "url": url_for('index')},
            {"title": "Login", "url": url_for('login')},
            {"title": "Register", "url": url_for('register')}
        ]
        return dict(nav_links=nav_links)


if __name__ == '__main__':
    app.run(debug=True)
