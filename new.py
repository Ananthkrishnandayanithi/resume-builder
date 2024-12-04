from flask import Flask, request, jsonify
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import docx2txt
from pdfminer.high_level import extract_text
import os
from werkzeug.utils import secure_filename
from langchain_google_genai.llms import GoogleGenerativeAI
import json
import re


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
app.config['OUTPUT_FOLDER'] = "output"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Hardcode your API key here
API_KEY = "AIzaSyBX_Gp607o2GndSOspHwmYhpIjELCiQ1MQ"

# Define prompts for each section
personal_details_prompt = PromptTemplate(
    input_variables=["content", "job_description"],
    template=""" 
    You are a professional resume builder. Use the extracted text below to create a well-formatted Personal Details section for a resume:

    Extracted Content:
    {content}

    Job Description:
    {job_description}

    Instructions:
    1. Extract only relevant personal details like full name, email, phone number, and location.
    2. Format these details professionally:
       - Name: Bold and prominent at the top.
       - Email, phone, and location: Clearly listed beneath the name.
    3. Exclude unnecessary details like marital status or hobbies unless explicitly relevant to the job description.
    4. If some personal details are missing, highlight them as "[MISSING]".

    Return the output as structured text for direct inclusion in a resume.
    """
)

skills_prompt = PromptTemplate(
    input_variables=["content", "job_description"],
    template=""" 
    You are a professional resume builder focusing on crafting the Technical Skills section based on the provided CV content and job description.

    Extracted Resume Content:
    {content}

    Job Description:
    {job_description}

    Instructions:
    1. Extract Technical Skills from the provided content and prioritize those relevant to the job description, if relevant skill is not available add one relevant skill.
    2. Organize skills into clear subcategories such as:
       - Programming Languages: Python, SQL
       - Frameworks & Libraries: React.js, TensorFlow, Keras
       - Tools & Technologies: REST APIs, Git, Docker
       - Databases: MySQL, PostgreSQL
       - Cloud Services: AWS, Azure
    3. Limit the total skills section to approximately 200 characters, focusing only on job-relevant skills.
    4. Exclude irrelevant or duplicate skills and use concise formatting for readability.
    5. Ensure the output is plain text and structured for direct inclusion in a resume.

    Return the Technical Skills section as a formatted bullet-point list with subcategories.
    """
)

experience_prompt = PromptTemplate(
    input_variables=["content", "job_description"],
    template=""" 
    You are an experienced professional resume builder. Below is the extracted Experience text and the job description:

    Extracted Content:
    {content}

    Job Description:
    {job_description}

    Instructions:
    1. Format the experience section clearly, including:
       - Job Title
       - Company Name
       - Employment Dates
       - Key Achievements (2-3 bullet points per role)
    2. Start each bullet with an action verb and quantify achievements (e.g., "Increased revenue by 20%").
    3. Use language that highlights impact, skills, and relevance to the job description.
    4. Exclude irrelevant experiences or details not aligned with the role.
    5. Maintain consistency in formatting across all roles.
    6. The experience segment should not exceed 500 characters. Highlight the relevant skill used in work experience.

    Output the formatted experience section for direct resume inclusion.
    """
)

projects_prompt = PromptTemplate(
    input_variables=["content", "job_description"],
    template=""" 
    You are a professional resume builder focusing on the Projects section of a resume. Use the extracted CV content and job description provided below:

    Extracted Resume Content:
    {content}

    Job Description:
    {job_description}

    Instructions:
    1. Identify key projects (minimum 2) from the content that align with the job description.
    2. For each project, include:
       - Project Title
       - Start and End Dates
       - Brief Description (50-80 characters)
       - Tools/Technologies Used
       - Key Achievement or Outcome (quantifiable where possible)
    """
)

education_prompt = PromptTemplate(
    input_variables=["content", "job_description"],
    template=""" 
    You are a professional resume builder. Below is the extracted Education text and the job description:

    Extracted Content:
    {content}

    Job Description:
    {job_description}

    Instructions:
    1. Format the education details in reverse chronological order.
    2. Include the following for each degree:
       - Degree Name
       - Institution Name
       - Graduation Date
       - GPA (if provided and relevant to the job)
    3. Highlight academic achievements or coursework relevant to the job description.
    4. Exclude irrelevant education or unnecessary details.

    Return the output as structured text formatted for inclusion in a resume.
    """
)
validation_prompts = {
    "Personal Details": PromptTemplate(
        input_variables=["extracted_content", "resume_content", "job_description"],
        template=""" 
        Response: {extracted_content}
        Resume: {resume_content}
        Job Description: {job_description}
    
### Instructions for Validation and Update:
1. *Cross-Check with Resume Context*:
    - Compare the generated JSON response with your provided resume details.
    - Validate that fields such as Full Name, Email, Phone, Location, Date of Birth, Languages, and LinkedIn match your resume text.

2. *Validation Rules*:
    - Check each field against your resume details to ensure correctness and consistency.
    - Ensure no field contains null, empty, or placeholder values.
    - Validate the format and structure of each field as follows:
        - *Full Name*: Matches the name in the resume. Ensure proper capitalization.
        - *Email*: Valid format, including "@" and a correct domain (e.g., ".com", ".org").
        - *Phone*: Includes a country code and is numeric. Correct any minor formatting issues (e.g., missing "+" or extra characters).
        - *Location*: Matches your resume location (City, Country). Correct capitalization if needed.
        - *Date of Birth*: Validate format (MM/DD/YYYY) if included.
        - *Languages*: Verify listed languages are relevant and match your resume.
        - *LinkedIn*: Ensure it is a valid URL starting with "https://www.linkedin.com/".

3. *Error Correction*:
    - If any field is invalid, not properly formatted, or not present in the resume, remove it entirely from the JSON.
    - Ensure consistency across all fields, especially capitalization, punctuation, and formatting.

4. *Final Output*:
    - Don't change the existing format.
        """
    ),
    "Skills": PromptTemplate(
        input_variables=["extracted_content", "resume_content", "job_description"],
        template=""" 
        response: {extracted_content}
        Resume: {resume_content}
        Job Description: {job_description}
        
        ### Instructions for Validating and Updating Skills:

1. *Cross-Check with Resume Content*:
    - Validate the skills in the generated JSON against the skills listed in the resume content.
    - Ensure that all skills are consistent with the resume's listed technologies, languages, frameworks, and tools.

2. *Validation Rules*:
    - *Languages*: Ensure the programming languages are valid and match those mentioned in the resume.
    - *Frameworks*: Verify that the frameworks or libraries listed are consistent with the resume.
    - *Technologies*: Cross-check if the technologies mentioned in the skills match the resume content (e.g., AWS, Docker, Azure).
    - *Tools*: Ensure tools and software such as Git, JIRA, and others are correctly listed.

3. *Skill Limit*:
    - Ensure that the total number of skills across all categories does not exceed *20*.
    - Each skill should be concise, and the total list of skills should be readable and ATS-friendly.

4. *Error Correction*:
    - Ensure no field contains null, placeholder values, or empty values.
    - If any skill is invalid or does not align with the resume content, remove it.
    - If a skill is missing but is relevant and implied from the resume (e.g., through related experience), include it.

5. *Final Output*:
    - Return the cleaned and updated skills section, formatted as a JSON object.
    - Ensure each skill is well-formatted, concise, and does not exceed *100 characters* including spaces.

6. *Formatting*:
    - Don't change the existing format.
    """
    )
}

# Function to extract text from uploaded file (PDF or DOCX)
def extract_text_from_upload(filepath):
    if filepath.endswith(".pdf"):
        return extract_text(filepath)
    elif filepath.endswith(".docx"):
        return docx2txt.process(filepath)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")

# Function to extract section names from LaTeX content
def extract_section_names(latex_content):
    """
    Extract section names from LaTeX content (i.e., titles of \section{...}).
    """
    section_names = re.findall(r'\\section\{(.*?)\}', latex_content)
    return section_names

def generate_and_validate_section(content, resume_content, job_description, section_name, api_key):
    chains = initialize_chains(api_key)
    
    # Generate section content from model
    chain = chains.get(section_name)
    if chain:
        generated_output = chain.run({"content": content, "job_description": job_description})
        
        # Print the generated output before validation
        print(f"Generated output for {section_name} before validation:\n{generated_output}")
        
        # Select the appropriate validation chain for the section
        validation_chain = create_chain(api_key, validation_prompts.get(section_name))
        
        if validation_chain:
            # Pass the generated content, resume, and job description to the validation chain
            validated_output = validation_chain.run({
                "extracted_content": generated_output,
                "resume_content": resume_content,
                "job_description": job_description
            })
            
            # Print the updated (validated) output after validation
            print(f"Updated output for {section_name} after validation:\n{validated_output}")
            return validated_output
        
    return f"Section {section_name} is not supported."


# Function to generate LaTeX section content
def generate_section_content(section, extracted_text, template, job_description, genai_model, api_key):
    """
    Generate LaTeX content for a specific section using the AI model.
    """
    llm = GoogleGenerativeAI(model=genai_model, api_key=api_key)
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=template.template,  # Extracting the template string from the PromptTemplate
            input_variables=["content", "job_description"]
        )
    )
    # Log the request to the model
    print(f"Generating section: {section}")
    print(f"Extracted Text: {extracted_text[:100]}...")  # Print only the first 100 characters for brevity
    print(f"Job Description: {job_description[:100]}...")  # Same for job description

    response = chain.run(content=extracted_text, job_description=job_description)

    # Log the response
    print(f"Model response for {section}: {response}")

    return f"\\section{{{section}}}\n{response}"

# Function to convert LaTeX content to plain text format
# Commenting this out as we now want LaTeX output
# def convert_latex_to_text(latex_content):
#     """
#     Convert the generated LaTeX content into a plain text formatted resume.
#     """
#     section_map = {
#         "personal_details": "Personal Details",
#         "skills": "Technical Skills",
#         "experience": "Work Experience",
#         "projects": "Projects",
#         "education": "Education"
#     }

#     # Remove LaTeX formatting and convert to plain text
#     lines = latex_content.split("\n")
#     plain_text = []

#     for line in lines:
#         # Remove LaTeX section headers
#         if line.startswith("\\section{"):
#             section = line.split("{")[1].split("}")[0].lower()
#             section_name = section_map.get(section, section.capitalize())
#             plain_text.append(f"\n{section_name}\n")
#         elif line.startswith("\\") or not line.strip():
#             continue
#         else:
#             plain_text.append(line.strip())

#     return "\n".join(plain_text)

@app.route('/generate_resume', methods=['POST'])
def generate_resume():
    try:
        # Parse request data
        uploaded_file = request.files.get("uploaded_file")
        job_description = request.form.get("job_description")

        # Validate required fields
        if not uploaded_file or not job_description:
            return jsonify({"error": "Missing required fields"}), 400

        # Secure the uploaded file name and save it
        filename = secure_filename(uploaded_file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(file_path)

        # Extract text from the uploaded file
        extracted_text = extract_text_from_upload(file_path)

        # Define a mapping for section names to their respective prompt templates
        prompt_templates = {
            "personal_details": personal_details_prompt,
            "skills": skills_prompt,
            "experience": experience_prompt,
            "projects": projects_prompt,
            "education": education_prompt
        }

        # Generate content for each section using the defined mapping
        results = {}
        for section, template in prompt_templates.items():
            generated_content = generate_section_content(
                section,
                extracted_text,
                template,
                job_description,
                "gemini-1.5-flash",
                API_KEY
            )
            results[section] = generated_content

        # Save the results to a LaTeX file for output
        output_file = os.path.join(app.config['OUTPUT_FOLDER'], "generated_resume.tex")
        with open(output_file, 'w') as f:
            for section, content in results.items():
                f.write(f"\n{content}\n")

        # Return a success response
        return jsonify({"message": "Resume generated successfully in LaTeX format", "file_path": output_file}), 200

    except Exception as e:
        print(f"Error generating resume: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

