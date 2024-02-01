from llm import GPT4QAModel
import json

model = GPT4QAModel()

def match_resumes_with_projects(resumes, project_descriptions):
    # Assuming resumes and project_descriptions are lists of strings
    allResumes = "Here is a list of resumes: \n"
    count = 1
    for resume in resumes:
        addThis = str(count) + ". " + resume + "\n"
        allResumes += addThis
        count += 1

    count = 1
    allProjects ="\nHere is a list of projects: \n"
    for project in project_descriptions:
        addThis = str(count) + ". " + project + "\n"
        allProjects += addThis
        count += 1
    prompt = allResumes + allProjects + "\nBased on these resumes and project descriptions, match each person to the best suited project."
    response = model.answer_question(prompt)
    print(prompt)
    print("ligma")
    #response = json.loads(response)
    return response



resumes = ['Name: Bob-I can code.', 'Name: Joe-I can manage projects.', 'Name: Lebron-I can clean.']
project_descriptions = ['Engineer', 'PM', 'janitor']
print(match_resumes_with_projects(resumes, project_descriptions))