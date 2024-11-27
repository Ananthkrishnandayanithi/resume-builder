from pdfminer.high_level import extract_text
import docx2txt
import google.generativeai as genai
import json
from stqdm import stqdm

SYSTEM_PROMPT = "You have to respond in JSON only. You are a smart assistant to career advisors at the Harvard Extension School. You will reply with JSON only."

CV_TEXT_PLACEHOLDER = "<CV_TEXT>"

SYSTEM_TAILORING = """
You are a smart assistant to career advisors at the Harvard Extension School. Your take is to rewrite
resumes to be more brief and convincing according to the Resumes and Cover Letters guide.
"""

TAILORING_PROMPT = """
Consider the following CV:
<CV_TEXT>

Your task is to rewrite the given CV. Follow these guidelines:
- Make sure the resume starts with the candidates name and role
- Be truthful and objective to the experience listed in the CV
- Be specific rather than general
- Rewrite job highlight items using STAR methodology (but do not mention STAR explicitly)
- Fix spelling and grammar errors
- Write to express not impress
- Articulate and don't be flowery
- Prefer active voice over passive voice
- Do not include a summary about the candidate

Improved CV:
"""

RESUME_TEMPLATE = """
-Make sure the resume starts with name and personal details section
-Make sure the second section contains details of skills and technologies
-Make sure the third section contains details of previous projects
-Make sure the fourth section contains details of previous/current experiences
-Make sure the fifth section contains details of education of the candidate
-Make sure the details are generated for a single page A4 pdf
-Make sure the resume meets the ATS quality
"""

BASICS_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface Basics {
    name: string;
    email: string;
    phone: string;
    website: string;
    address: string;
}

Write the basics section according to the Basic schema. On the response, include only the JSON.
"""

EDUCATION_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface EducationItem {
    institution: string;
    area: string;
    additionalAreas: string[];
    studyType: string;
    startDate: string;
    endDate: string;
    score: string;
    location: string;
}

interface Education {
    education: EducationItem[];
}

Write the education section according to the Education schema. On the response, include only the JSON.
"""

AWARDS_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface AwardItem {
    title: string;
    date: string;
    awarder: string;
    summary: string;
}

interface Awards {
    awards: AwardItem[];
}

Write the awards section according to the Awards schema. Include only the awards section. On the response, include only the JSON.
"""

PROJECTS_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface ProjectItem {
    name: string;
    description: string;
    keywords: string[];
    url: string;
}

interface Projects {
    projects: ProjectItem[];
}

Write the projects section according to the Projects schema. Include all projects, but only the ones present in the CV. On the response, include only the JSON.
"""

SKILLS_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

type HardSkills = "Programming Languages" | "Tools" | "Frameworks" | "Computer Proficiency";
type SoftSkills = "Team Work" | "Communication" | "Leadership" | "Problem Solving" | "Creativity";
type OtherSkills = string;

Now consider the following TypeScript Interface for the JSON schema:

interface SkillItem {
    name: HardSkills | SoftSkills | OtherSkills;
    keywords: string[];
}

interface Skills {
    skills: SkillItem[];
}

Write the skills section according to the Skills schema. Include only up to the top 4 skill names that are present in the CV and related with the education and work experience. On the response, include only the JSON.
"""

WORK_PROMPT = """
You are going to write a JSON resume section for an applicant applying for job posts.

Consider the following CV:
<CV_TEXT>

Now consider the following TypeScript Interface for the JSON schema:

interface WorkItem {
    company: string;
    position: string;
    startDate: string;
    endDate: string;
    location: string;
    highlights: string[];
}

interface Work {
    work: WorkItem[];
}

Write a work section for the candidate according to the Work schema. Include only the work experience and not the project experience. For each work experience, provide a company name, position name, start and end date, and bullet point for the highlights. Follow the Harvard Extension School Resume guidelines and phrase the highlights with the STAR methodology
"""

def extract_text_from_pdf(file):
    return extract_text(file)


def extract_text_from_docx(file):
    return docx2txt.process(file)


def extract_text_from_upload(file):
    if file.endswith(".pdf"):
        text = extract_text_from_pdf(file)
        return text
    elif file.endswith(".docx"):
        text = extract_text_from_docx(file)
        return text
    elif file.endswith("json"):
        return file.getvalue().decode("utf-8")
    else:
        return file.getvalue().decode("utf-8")

def generate_section_content(section_name, extracted_text, template, job_description, model, api_key):
    """
    Generate tailored LaTeX content for a section using the LLM.
    """
    prompt = f"""
    Generate a LaTeX formatted resume section for {section_name} based on the following:
    - Extracted content: {extracted_text}
    - Section template: {template}
    - Job description: {job_description}
    - Each point should not exceed 80 letters
    - Make sure that the the details filled in the template without changing any syntax and the number of bullet points should always be same
    - The details should be modified enhanced to match 100 percent to the job description with little bluffing and adding additional content
    - Each data should not be precisely within a single line
    The output should only include LaTeX code for the section, formatted professionally.
    """
    genai.configure(api_key=api_key)
    model_instance = genai.GenerativeModel(model)
    response = model_instance.generate_content(prompt)
    return response.text if response else ""


def tailor_resume(cv_text, api_key, model, job_description):
    """Tailor a resume using Google's Generative AI"""
    try:
        # Configure the generative AI with the provided API key
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)

        # Prepare the tailored prompt with section prioritization
        combined_prompt = f"""
        
        Using the provided job description and resume content, generate a revised resume that is fully optimized for the specific role. The tailored resume should:

        1. First, display personal details (name, contact, LinkedIn) followed by a "Technologies" section showcasing the relevant skills. This should appear first inside the header.
        2. Then, present the "Experience" section, followed by any projects listed in the CV.
        3. Ensure that sections such as Skills and Experience are formatted correctly in LaTeX.
        4. Highlight skills, achievements, and experiences directly relevant to the job description.
        5. Use language aligned with the keywords present in the job description to improve ATS compatibility.
        6. Include hypothetical or inferred project examples that showcase the application of required skills and technologies.
        7. Avoid providing templates or generic instructions. Ensure professional formatting and accurate grammar.
        8. Ensure the final output is concise, polished, and immediately usable for the job application.
        
        Job Description and required qualifications:
        {job_description}

        Resume Template:
        {RESUME_TEMPLATE}

        Original Resume Content:
        {cv_text}
        """

        # Generate tailored resume content
        response = model_instance.generate_content(combined_prompt)
        return response.text
    except Exception as e:
        print(f"Failed to tailor resume: {e}")
        return cv_text
