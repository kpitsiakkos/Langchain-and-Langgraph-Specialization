import os
import re
import openai
import pandas as pd
from pypdf import PdfReader
from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI  # updated import

# Load environment variables
load_dotenv(find_dotenv())
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Constants ---
EXTRACTION_TEMPLATE = """
Extract the following fields from the invoice text below:
Invoice ID, DESCRIPTION, Issue Date, UNIT PRICE, AMOUNT, Bill For, From, Terms.

Rules:
- Remove any currency symbols (e.g. $)
- Return only a Python dictionary, no extra text

Invoice text:
{pages}

Expected output format:
{{'Invoice ID': '1001329', 'DESCRIPTION': 'Web Design', 'Issue Date': '5/4/2023',
  'UNIT PRICE': '550.00', 'AMOUNT': '1100.00', 'Bill For': 'James',
  'From': 'Excel Company', 'Terms': 'Net 30'}}
"""

DF_SCHEMA = {
    "Invoice ID": pd.Series(dtype="str"),
    "DESCRIPTION": pd.Series(dtype="str"),
    "Issue Date": pd.Series(dtype="str"),
    "UNIT PRICE": pd.Series(dtype="str"),
    "AMOUNT": pd.Series(dtype="str"),
    "Bill For": pd.Series(dtype="str"),
    "From": pd.Series(dtype="str"),
    "Terms": pd.Series(dtype="str"),
}


def get_pdf_text(pdf_doc) -> str:
    """Extract raw text from all pages of a PDF file."""
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text


def extract_data_with_llm(pages_data: str) -> str:
    """Send extracted PDF text to the LLM and return structured response."""
    prompt_template = PromptTemplate(
        input_variables=["pages"],
        template=EXTRACTION_TEMPLATE
    )
    llm = OpenAI(temperature=0.2)  # lower temp for more consistent extraction
    return llm.invoke(prompt_template.format(pages=pages_data))


def parse_llm_response(llm_response: str) -> dict | None:
    """Parse the LLM's response and extract a Python dictionary."""
    pattern = r'\{(.+?)\}'
    match = re.search(pattern, llm_response, re.DOTALL)
    if not match:
        print("Warning: No dictionary found in LLM response.")
        return None
    try:
        return eval("{" + match.group(1) + "}")
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        return None


def create_docs(user_pdf_list) -> pd.DataFrame:
    """Process a list of PDF files and return a DataFrame of extracted invoice data."""
    df = pd.DataFrame(DF_SCHEMA)

    for pdf_file in user_pdf_list:
        print(f"Processing: {pdf_file}")

        raw_text = get_pdf_text(pdf_file)
        llm_response = extract_data_with_llm(raw_text)
        data_dict = parse_llm_response(llm_response)

        if data_dict:
            df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
            print(f"Extracted: {data_dict}")
        else:
            print(f"Skipped: {pdf_file} — could not parse response.")

    print("All files processed.")
    return df