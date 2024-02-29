from flask import Flask, render_template, request, redirect, url_for, session, flash
from util import extract_text_from_pdf, allowed_file, summarize_resume, get_embedding_from_resume, get_embedding_from_project, get_employee_embedding, cos_similarity, get_embedding_sum
from llm import GPT4QAModel
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import numpy as np

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.secret_key = 'your_secret_key'

db = SQLAlchemy(app)


#embedding_sum = np.zeros(768)
#num_embeddings = 0


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(80), nullable=False)
    embedding_sum = db.Column(db.String(15000)) 
    num_embeddings = db.Column(db.Integer)

class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100))  
    summary = db.Column(db.Text, nullable=False) 
    skills = db.Column(db.Text, nullable=False)  
    hobbies = db.Column(db.Text, nullable=False)
    jobs = db.Column(db.Text, nullable=False)
    embedding = db.Column(db.String(15000)) 
    user = db.relationship('User', backref=db.backref('employees', lazy=True))
    

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
    embedding = db.Column(db.String(15000))
    best_employee_id = db.Column(db.Integer, db.ForeignKey('employee.id'))
    best_employee_name = db.Column(db.String(100)) 
    user = db.relationship('User', backref=db.backref('projects', lazy=True))

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
        num_embeddings = 0
        embedding_sum = np.zeros(768)
        string_embedding_sum = np.array2string(embedding_sum)
        new_user = User(username=username, password=hashed_password, email=email, embedding_sum = string_embedding_sum, num_embeddings = num_embeddings)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for('dashboard'))
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

def update_best_employees(employee):
    model = GPT4QAModel()
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()
        if len(user.employees) == 1:
            for project in user.projects:
                project.best_employee_id = employee.id
                project.best_employee_name = employee.name
        
        else:
            for project in user.projects:
                project_embedding = np.fromstring(project.embedding[1:-1], sep=' ')
                curr_best_employee = Employee.query.get(project.best_employee_id)
                best_employee_prompt = "Here is some information about the best employee so far: \n" + makeEmployeePrompt(curr_best_employee) + "\n\n"
                curr_employee_prompt = "Now, here is some information about the new employee we are evaluating: \n"+ makeEmployeePrompt(employee) + "\n\n"
                prompt = "Imagine you are a recruiter and you want to hire the best talent possible. You are looking at the best employee currently and a new employee applying for the postion. The new employee may or may not be better than the curent best employee.\n"
                proj_prompt = "Here is some information about the project:\nProject Title: "+project.title + "\nProject Description: "+project.description + "\n\n"
                end = "Looking at the current best employee and new employee applying, is the new employee better suited for this project compared to the current best employee? Output either \"Yes\" or \"No\". "
                
                fullprompt = prompt + proj_prompt + best_employee_prompt + curr_employee_prompt + end

                res = model.answer_question(fullprompt)
                print(employee.name, "Is this guy better?:", res)
                if "yes" in res.lower(): #replace
                    print("inhere\n")
                    project.best_employee_id = employee.id
                    project.best_employee_name = employee.name
                
                
        db.session.commit()
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
            curr_emb_sum = get_embedding_sum(user)
            new_embedding_sum =  curr_emb_sum + embedding
            user.embedding_sum = np.array2string(new_embedding_sum)
            user.num_embeddings += 1

            db.session.add(new_employee)
            db.session.commit()
            update_best_employees(new_employee)
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

    user = User.query.filter_by(username=session['username']).first()
    employee_to_delete = Employee.query.get(employee_id)
    if employee_to_delete:
        emp_embedding = get_employee_embedding(employee_to_delete)
        curr_emb_sum = get_embedding_sum(user)
        new_embedding_sum = curr_emb_sum - emp_embedding
        user.embedding_sum = np.array2string(new_embedding_sum)

        user.num_embeddings -= 1
        db.session.delete(employee_to_delete)
        db.session.commit()
        flash('Employee deleted successfully')
    else:
        flash('Employee not found')

    return redirect(url_for('dashboard'))

def get_best_employee_id_name_for_project(embedding):
  best_similarity = -1
  best_employee= None
  if 'username' in session:
    user = User.query.filter_by(username=session['username']).first()
    #project = Project.query.get(project_id)
    for employee in user.employees:
        employee_embedding = get_employee_embedding(employee)
        if employee_embedding is not None:
            similarity = cos_similarity(embedding, employee_embedding)
            print(employee.name, f'{similarity=}')
            if similarity > best_similarity:
                best_similarity = similarity
                best_employee= employee
    return best_employee.id, best_employee.name
  else:
    flash('User not logged in')
    return redirect(url_for('login'))

def get_5_best_employees_for_project(embedding):
  best_similarity = -1
  best_employee= None
  if 'username' in session:
    user = User.query.filter_by(username=session['username']).first()
    #project = Project.query.get(project_id)
    similarity_scores = []
    for employee in user.employees:
        employee_embedding = get_employee_embedding(employee)
        if employee_embedding is not None:
            similarity = cos_similarity(embedding - get_embedding_sum(user) / user.num_embeddings, employee_embedding - get_embedding_sum(user) / user.num_embeddings)
            print(employee.name, f'{similarity=}')
            similarity_scores.append((employee, similarity))
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            similarity_scores = similarity_scores[:5]
    best_employees = []
    for item in similarity_scores:
        best_employees.append(item[0])

    return best_employees
  else:
    flash('User not logged in')
    return redirect(url_for('login'))


def makeEmployeePrompt(employee):
    temp1 = "Name: " + employee.name
    temp2 = "\nID: " + str(employee.id)
    temp3 = "\nUser ID: " + str(employee.user_id)
    temp4 = "\nSummary: " + employee.summary
    temp5= "\nHobbies: " + ", ".join(employee.hobbies)
    temp6 = "\nJobs: " + ", ".join(employee.jobs)
    temp7 = "\Skills: " + ", ".join(employee.skills)
    return temp1+temp2+temp3+temp4+temp5+temp6 +temp7


def llm_get_best_employee_id_name_for_project(best_employees, new_project):
    model = GPT4QAModel()
    best_employee = None
    first = True
    for employee in best_employees:
        if first: #set best employee to first one
            best_employee = employee
            first = False
        else:
            best_employee_prompt = "Here is some information about the best employee so far: \n" + makeEmployeePrompt(best_employee) + "\n\n"
            curr_employee_prompt = "Now, here is some information about the new employee we are evaluating: \n"+ makeEmployeePrompt(employee) + "\n\n"
            prompt = "Imagine you are a recruiter and you want to hire the best talent possible. You are looking at the best employee currently and a new employee applying for the postion. The new employee may or may not be better than the curent best employee.\n"
            proj_prompt = "Here is some information about the project:\nProject Title: "+new_project.title + "\nProject Description: "+new_project.description + "\n\n"
            end = "Looking at the current best employee and new employee applying, is the new employee better suited for this project compared to the current best employee? Output either \"Yes\" or \"No\". "


            fullprompt = prompt + proj_prompt + best_employee_prompt + curr_employee_prompt + end

            res = model.answer_question(fullprompt)
            print(employee.name, "Is this guy better?:", res)
            if "yes" in res.lower(): #replace
                print("inhere\n")
                best_employee = employee

    return best_employee

@app.route('/add_project', methods=['POST'])
def add_project():
    if 'username' in session:
        user = User.query.filter_by(username=session['username']).first()
        title = request.form.get('title')
        description = request.form.get('description')
        embedding = get_embedding_from_project(description)
        strembedding = np.array2string(embedding)
        
        curr_emb_sum = get_embedding_sum(user)
        new_embedding_sum = curr_emb_sum + embedding
        user.embedding_sum = np.array2string(new_embedding_sum)
        user.num_embeddings += 1
        new_project = Project(user_id=user.id, title=title, description=description, embedding=strembedding)
        best_employees = get_5_best_employees_for_project(embedding)
        if len(user.employees) > 0:
            best_employee = llm_get_best_employee_id_name_for_project(best_employees, new_project)
            new_best_employee_id, new_best_employee_name = best_employee.id, best_employee.name
            new_project.best_employee_id = new_best_employee_id
            new_project.best_employee_name = new_best_employee_name
        #print(new_best_employee_id)
        #print(new_best_employee_name)
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
