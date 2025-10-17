import os
from openai import OpenAI
from PyPDF2 import PdfReader
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
client = OpenAI(
            api_key=os.environ.get("API_KEY"))


class InterviewLOGIC:


    def extract_text_from_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)

        # Initialize an empty string to store text
        text = ""

        # Loop through each page in the PDF
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text


    def generate_question(self, resume_text, company, position, qa_pairs=[]):

        prompt = f"""
        You are now an HR interviewer for a reputable company.
        Your task is to conduct a realistic and professional job interview with the user.
        
        Here is the candidate’s resume:

        {resume_text}
        
        Here are the details of the job they are applying for:
        Company: {company}
        Position: {position}
        
        Interview:
        {self.QA_summary(qa_pairs) if qa_pairs else ""}

        Based on it, ask one relevant interview question.
        Adjust the tone and difficulty of questions depending on the job level (intern, junior, senior, manager).
        Ask one question at a time.
        Stay in character as an HR professional.
        The question should be natural and conversational.
        """
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    
    def generate_conclusion(self,resume_text, qa_pairs):
        prompt = f"""
        You are an HR interviewer. Based on the following candidate resume and their interview Q&A,
        give a short, professional conclusion or evaluation (3–5 sentences). Mention strengths and possible weaknesses.

        Resume:
        {resume_text}

        Interview:
        {self.QA_summary(qa_pairs) if qa_pairs else ""}
        
        in the end, tell the candidate whether they got the job or not and why.
        Make sure to seperate decision part in anoter paragraph. 
        """
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()

    def QA_summary(self, QA_pairs):
        summary = "\n".join([f"Q: {q}\nA: {a}" for q, a in QA_pairs])
        return summary