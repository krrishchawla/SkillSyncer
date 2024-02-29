from flask import Flask, render_template, request, redirect, url_for, session, flash
from util import extract_text_from_pdf, allowed_file, summarize_resume, get_embedding_from_resume, get_embedding_from_project, get_employee_embedding, cos_similarity, update_best_employees, update_best_employees_llm_actuallyupdate, update_best_employees_llm, get_best_employee_id_name_for_project, makeEmployeePrompt, llm_get_best_employee_id_name_for_project
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np
from llm import GPT4QAModel
from models import db, User, Employee, Project

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.secret_key = 'your_secret_key'
db.init_app(app)


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

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()  # Clear the session
    flash('You have been logged out.')
    return redirect(url_for('login'))

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

@app.route('/dashboard', methods=["GET"])
def dashboard():
    current_date = datetime.now()
    formatted_date = str(current_date.strftime("%B %d, %Y"))
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()
        if user:
            employees = Employee.query.filter_by(user_id=user.id).all()
            projects = Project.query.filter_by(user_id=user.id).all()
            employees_dict = {}
            for employee in employees:
                employees_dict[employee] = employee.jobs.strip("[]").split(',')[0].strip("'")
            return render_template('dashboard.html', username=session['username'], date=formatted_date, employees=reversed(employees), projects=reversed(projects), employees_dict=employees_dict)
        else:
            flash('User not found')
            return redirect(url_for('login'))
    else:
        flash('User not logged in')
        return redirect(url_for('login'))


@app.route('/add_employee', methods=['POST'])
def add_employee():
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()
        resume_file = request.files.get('resume')
        if resume_file and allowed_file(resume_file.filename):
            resume_text = extract_text_from_pdf(resume_file)
            name, summary, skills, hobbies, jobs = summarize_resume(resume_text)
            embedding = get_embedding_from_resume(resume_text)
            strembedding = np.array2string(embedding)
            new_employee = Employee(user_id=user.id, name=name, summary=summary, embedding = strembedding, skills=skills, hobbies=hobbies, jobs=jobs)
            db.session.add(new_employee)
            db.session.commit()
            update_best_employees_llm(new_employee) #needa look into projects here
            flash('Employee added successfully')
        else:
            flash('Invalid file format or no file uploaded')
        return redirect(url_for('dashboard'))
    else:
        flash('User not logged in')
        return redirect(url_for('login'))

@app.route('/delete_employee/<int:employee_id>', methods=['POST'])
def delete_employee(employee_id):
    if 'username' not in session:
        return redirect(url_for('login'))

    employee_to_delete = Employee.query.get(employee_id)
    if employee_to_delete:
        db.session.delete(employee_to_delete)
        db.session.commit()
        flash('Employee deleted successfully')
    else:
        flash('Employee not found')

    return redirect(url_for('dashboard'))


@app.route('/add_project', methods=['POST'])
def add_project():
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()
        title = request.form.get('title')
        description = request.form.get('description')
        #embedding = get_embedding_from_project(description)
        #strembedding = np.array2string(embedding)

        new_project = Project(user_id=user.id, title=title, description=description) #embedding=strembedding)
        best_employee, reason = llm_get_best_employee_id_name_for_project(new_project)

        new_project.best_employee_reason=reason

        new_best_employee_id, new_best_employee_name = best_employee.id, best_employee.name
        new_project.best_employee_id = new_best_employee_id
        new_project.best_employee_name = new_best_employee_name
        db.session.add(new_project)

        db.session.commit()

        flash('Project added successfully')
        return redirect(url_for('dashboard'))
    else:
        flash('User not logged in')
        return redirect(url_for('login'))

@app.route('/delete_project/<int:project_id>', methods=['POST'])
def delete_project(project_id):
    if 'username' not in session:
        flash('User not logged in')
        return redirect(url_for('login'))

    project_to_delete = Project.query.get(project_id)
    if project_to_delete:
        db.session.delete(project_to_delete)
        db.session.commit()
        flash('Project deleted successfully')
    else:
        flash('Project not found')

    return redirect(url_for('dashboard'))


if __name__ == '__main__':
    app.run(debug=True)