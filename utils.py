from pdfminer.high_level import extract_text
import docx2txt
import google.generativeai as genai
from stqdm import stqdm
import json

# Constants
SYSTEM_PROMPT = "You have to respond in JSON only. You are a smart assistant to career advisors at the Harvard Extension School. You will reply with JSON only."
RESUME_TEMPLATE = """
-Single A4 page with all sections fitting within the page.
-Make sure the resume starts with name and personal details section.
-Make sure the second section contains details of skills and technologies.
-Make sure the third section contains details of previous projects.
-Make sure the fourth section contains details of previous/current experiences.
-Make sure the fifth section contains details of education of the candidate.
-Make sure the details are generated for a single-page A4 pdf.
-Make sure the resume meets ATS quality.
"""

# File extraction
def extract_text_from_pdf(file):
    try:
        return extract_text(file)
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file):
    try:
        return docx2txt.process(file)
    except Exception as e:
        print(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_upload(file):
    try:
        if file.endswith(".pdf"):
            return extract_text_from_pdf(file)
        elif file.endswith(".docx"):
            return extract_text_from_docx(file)
        elif file.endswith(".json"):
            return file.getvalue().decode("utf-8")
        else:
            return file.getvalue().decode("utf-8")
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

# Resume tailoring logic
def generate_section_content(section_name, extracted_text, template, job_description, model, api_key):
    """
    Generate tailored content for a resume section using the Generative AI model.
    """
    try:
        prompt = f"""
        Generate a resume section for {section_name} based on the following:
        - Extracted content: {extracted_text}
        - Section template: {template}
        - Job description: {job_description}
        - Each point should not exceed 100 characters.
        - Ensure that all details fit within a single A4 page.
        - Each section must follow the provided template format.
        """
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(prompt)
        return response.text if response else "No content generated."
    except Exception as e:
        
        return "Error generating section content."

def tailor_resume(cv_text, api_key, model, job_description):
    """Tailor a resume using Google's Generative AI."""
    try:
        # Configure Generative AI
        genai.configure(api_key=api_key)
        model_instance = genai.GenerativeModel(model)

        # Prepare the prompt
        prompt = f"""
        Using the provided job description and resume content, generate a revised resume that is fully optimized for the specific role. Ensure the resume:
        1. Fits a single A4 page with all required sections.
        2. Includes personal details, skills, projects, experience, and education sections.
        3. Uses language aligned with keywords in the job description for ATS optimization.
        4. Highlights relevant achievements using the STAR methodology without explicitly naming it.
        5. Avoids generic instructions and maintains professional formatting and accurate grammar.

        Job Description: {job_description}
        Resume Template: {RESUME_TEMPLATE}
        Original Resume Content: {cv_text}
        """

        # Generate the tailored resume
        response = model_instance.generate_content(prompt)
        return response.text if response else "Failed to generate tailored resume."
    except Exception as e:
       
        return "Error generating tailored resume."

# Example usage
if __name__ == "__main__":
    # Simulated inputs
    cv_text = "<Your CV Text Here>"
    api_key = "<Your_API_Key>"
    model = "<Model_Name>"
    job_description = "<Job Description>"
    


    tailored_resume = tailor_resume(cv_text, api_key, model, job_description)
    print(tailored_resume)
