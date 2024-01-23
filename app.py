import asyncio
from flask import Flask, render_template, request, redirect, url_for, session
from PIL import Image
import numpy as np
from clip import get_probs
import json
from util import final_response
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.secret_key = 'your_secret_key'

db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(80), nullable=False)

# Function to initialize the database
def initialize_db():
    with app.app_context():
        db.create_all()

# Call the function to initialize the database
initialize_db()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/explore_form')
def explore_form():
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session['username'] = user.username
            return redirect(url_for('dashboard'))
        else:
            return render_template('login.html', error="Invalid username or password")
    return render_template('login.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        hashed_password = generate_password_hash(password)  # Default hashing method

        new_user = User(username=username, password=hashed_password, email=email)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/show-users')
def show_users():
    users = User.query.all()
    user_data = '<br>'.join([f'Username: {user.username}, Email: {user.email}' for user in users])
    return user_data


@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)



# @app.route('/explore_encryption')
# def explore_encryption():
#     return render_template('encrypt.html')


# def generate_categories(pests):
#     return [f"a photo of the pest {x}" for x in pests]

# @app.route('/process_form', methods=['POST'])
# def process_form():
#     if 'image' not in request.files or request.files['image'].filename == "":
#         return "No image uploaded!", 400  # Return an error message

#     file = request.files['image']

#     try:
#         # Ensure the file is an image
#         image = Image.open(file.stream)
#         image.save("./img/img_1.png")
#     except IOError:
#         return "Invalid image file!", 400  # Return an error message

#     # Retrieve text input from the form
#     crop = request.form.get('crop_name', '')
#     state = request.form.get('state', '')

#     with open("pests.json", "r") as file:
#         pests = json.load(file)

#     dict_probs = get_probs(categories=generate_categories(pests))
#     sorted_dict_probs = dict(sorted(dict_probs.items(), key=lambda item: item[1]))

#     # Assuming you want the last item after sorting
#     if sorted_dict_probs:
#         pest, percentage = list(sorted_dict_probs.items())[-1]
#         pest = pest.replace("a photo of the pest ", "")
#         answer = final_response(crop=crop, state=state, pest=pest, topk=3)

#         # Ensure answer is a dictionary
#         if isinstance(answer, str):
#             output = json.loads(answer)
#         else:
#             output = answer

#         # Load URLs from pesticide_urls.json
#         with open("pesticide_urls.json", "r") as url_file:
#             pesticide_urls = json.load(url_file)

#         # Add URLs to the output data
#         for pesticide_number in output:
#             if pesticide_number in pesticide_urls:
#                 output[pesticide_number]['url'] = pesticide_urls[pesticide_number]
#             else:
#                 output[pesticide_number]['url'] = None  # Or a default value

#     else:
#         output = "No pest data found"

#     return render_template('identification.html', dict_probs=output, pest=pest, percentage=percentage, crop=crop, location=state)


