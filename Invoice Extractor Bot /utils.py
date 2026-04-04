import os
import replicate
import pandas as pd
import json
import re
from pypdf import PdfReader
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=dotenv_path)

client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extracted_data(pages_data):
    prompt = f"""Extract the following fields from this invoice text and return ONLY a JSON object, no explanation, no markdown:
invoice no., Description, Quantity, Date, Unit price, Amount, Total, Email, Phone number, Address.

Invoice text:
{pages_data}

Return exactly this structure (use empty string if a field is not found, strip currency symbols):
{{"Invoice no.": "", "Description": "", "Quantity": "", "Date": "", "Unit price": "", "Amount": "", "Total": "", "Email": "", "Phone number": "", "Address": ""}}"""

    output = client.run(
        "meta/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
        input={
            "prompt": prompt,
            "temperature": 0.1,
            "top_p": 0.9,
            "max_length": 512,
            "repetition_penalty": 1
        }
    )

    full_response = "".join(output)

    try:
        full_response = full_response.replace("```json", "").replace("```", "").strip()
        match = re.search(r'\{.+\}', full_response, re.DOTALL)
        if match:
            data_dict = json.loads(match.group())
        else:
            data_dict = {}
    except json.JSONDecodeError:
        data_dict = {}

    return data_dict


def create_docs(user_pdf_list):
    df = pd.DataFrame({
        "Invoice no.": pd.Series(dtype="str"),
        "Description": pd.Series(dtype="str"),
        "Quantity": pd.Series(dtype="str"),
        "Date": pd.Series(dtype="str"),
        "Unit price": pd.Series(dtype="str"),
        "Amount": pd.Series(dtype="str"),
        "Total": pd.Series(dtype="str"),
        "Email": pd.Series(dtype="str"),
        "Phone number": pd.Series(dtype="str"),
        "Address": pd.Series(dtype="str"),
    })

    for pdf_file in user_pdf_list:
        print(f"Processing: {pdf_file}")
        raw_text = get_pdf_text(pdf_file)
        print("Text extracted, sending to LLaMA via Replicate...")

        data_dict = extracted_data(raw_text)
        print(f"Extracted: {data_dict}")

        df = pd.concat([df, pd.DataFrame([data_dict])], ignore_index=True)
        print("Done.")

    return df