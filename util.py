from tenacity import retry, stop_after_attempt, wait_random_exponential
from llm import GPT4QAModel
from PyPDF2 import PdfReader
import numpy as np
import os
import json
from json.decoder import JSONDecodeError
from tqdm import tqdm
from PyPDF2.errors import PdfReadError
from InstructorEmbedding import INSTRUCTOR


myModel = INSTRUCTOR('hkunlp/instructor-xl')


def get_employee_embedding(employee):
  if employee:
    employee_embedding = np.fromstring(employee.embedding[1:-1], sep=' ')  # Convert the string back to a NumPy array
    return employee_embedding
  return None

def get_embedding_sum(user):
  if user:
    embedding_sum = np.fromstring(user.embedding_sum[1:-1], sep=' ')  # Convert the string back to a NumPy array
    return embedding_sum
  return None

def get_project_embedding(project):
  if project:
    proj_embedding = np.fromstring(project.embedding[1:-1], sep=' ')  # Convert the string back to a NumPy array
    return proj_embedding
  return None



def get_embedding_from_resume(resume_text):
  resume_instruction = [["Represent the employee resume document for retrieving suitable projects: ",resume_text]]
  embedding_resume = myModel.encode(resume_instruction)
  return embedding_resume[0]

def get_embedding_from_project(project_text):
  project_instruction = [["Represent the project description for retrieval: ", project_text]]
  embedding_project = myModel.encode(project_instruction)
  return embedding_project[0]

def cos_similarity(embedding_resume, embedding_project):
  similarity = np.dot(embedding_resume, embedding_project) / (np.linalg.norm(embedding_resume) * np.linalg.norm(embedding_project))
  return similarity

def extract_text_from_pdf(pdf_file):
    pdf_text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    return pdf_text

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'pdf'}

def summarize_resume(resume_text):
    # Change later
    prompt = '''You are a data retriever. I will give you a resume, and I want you to summarize it.
    I want you to output a python dictionary, with the keys - name, summary, skills, hobbies, jobs.
    For name, extract the name and add it.
    For summary, give me a short 100 word summary of the whole resume and the person.
    For skills, list out skills they are good at.
    For hobbies, list of hobbies if they have, or else just write None.
    For jobs, list of 5 job titles that you think this person may be suitable for.
    Here is the resume:
    '''
    print(f"Resume Text: {resume_text}")
    prompt += resume_text
    model = GPT4QAModel()
    response = model.answer_question(prompt)
    print(f"response: {response}")
    response = json.loads(response)
    name = str(response['name'])
    summary = str(response['summary'])
    skills = str(response['skills'])
    hobbies = str(response['hobbies'])
    jobs = str(response['jobs'])
    return name, summary, skills, hobbies, jobs



def main():
    prompt = "Hello, world!"
    model = GPT4QAModel()
    response = model.answer_question(prompt)
    print(f"response: {response}")
    # pass

if __name__ == "__main__":
    main()