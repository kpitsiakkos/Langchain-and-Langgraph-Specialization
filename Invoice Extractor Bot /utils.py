import os
import time
import re
import replicate
import pandas as pd
import json
from pypdf import PdfReader
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(dotenv_path=dotenv_path)

client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

COLUMNS = [
    "Invoice No.", "Invoice Date", "Due Date", "PO Number", "Payment Terms",
    "Vendor Name", "Vendor Address", "Vendor Email", "Vendor Phone",
    "Bill To Name", "Bill To Contact", "Bill To Address", "Bill To Email", "Bill To Phone",
    "Descriptions", "Quantities", "Unit Prices", "Discounts", "Amounts",
    "Subtotal", "Discount Total", "Tax Rate", "Tax Amount", "Shipping", "Total Due",
    "Bank Name", "Account No.", "Routing No.", "SWIFT/BIC", "IBAN"
]


def get_pdf_text(pdf_doc):
    text = ""
    pdf_reader = PdfReader(pdf_doc)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def run_llama(prompt):
    output = client.run(
        "meta/llama-2-70b-chat:2c1608e18606fad2812020dc541930f2d0495ce32eee50074220b87300bc16e1",
        input={"prompt": prompt, "temperature": 0.1, "top_p": 0.9,
               "max_new_tokens": 700, "repetition_penalty": 1}
    )
    return "".join(output)


def parse_json(text):
    text = text.strip()
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return {}
    fragment = match.group()
    try:
        return json.loads(fragment)
    except json.JSONDecodeError:
        pass
    try:
        closed = re.sub(r',?\s*"[^"]*"\s*:\s*"[^"]*$', '', fragment).rstrip(',') + '}'
        return json.loads(closed)
    except json.JSONDecodeError:
        return {}


def extract_header_and_financials(text):
    prompt = f"""Return ONLY a JSON object with these fields from the invoice. No preamble, no explanation.

Invoice:
{text}

JSON:
{{"Invoice No.": "", "Invoice Date": "", "Due Date": "", "PO Number": "", "Payment Terms": "", "Vendor Name": "", "Vendor Address": "", "Vendor Email": "", "Vendor Phone": "", "Bill To Name": "", "Bill To Contact": "", "Bill To Address": "", "Bill To Email": "", "Bill To Phone": "", "Subtotal": "", "Discount Total": "", "Tax Rate": "", "Tax Amount": "", "Shipping": "", "Total Due": "", "Bank Name": "", "Account No.": "", "Routing No.": "", "SWIFT/BIC": "", "IBAN": ""}}"""
    response = run_llama(prompt)
    print(f"Header+Financials: {response[:300]}")
    return parse_json(response)


def parse_lineitems_from_text(text):
    """Parse line items directly from PDF text using regex — no LLaMA needed."""
    descriptions, quantities, unit_prices, discounts, amounts = [], [], [], [], []

    pattern = re.compile(
        r'\n(\d+)\n'
        r'((?:.+\n)*?)'
        r'(\d+(?:\.\d+)?)\n'
        r'\$([\d,]+\.?\d*)\n'
        r'(\d+%)\n'
        r'\$([\d,]+\.?\d*)',
        re.MULTILINE
    )

    for m in pattern.finditer(text):
        desc = m.group(2).replace('\n', ' ').strip()
        desc = re.sub(r'\(.*?\)', '', desc).strip()
        descriptions.append(desc)
        quantities.append(m.group(3))
        unit_prices.append(m.group(4))
        discounts.append(m.group(5))
        amounts.append(m.group(6))

    print(f"Parsed {len(descriptions)} line items from text")

    return {
        "Descriptions": " | ".join(descriptions),
        "Quantities":   " | ".join(quantities),
        "Unit Prices":  " | ".join(unit_prices),
        "Discounts":    " | ".join(discounts),
        "Amounts":      " | ".join(amounts),
    }


def extracted_data(pages_data):
    header = extract_header_and_financials(pages_data)
    items  = parse_lineitems_from_text(pages_data)

    combined = {**header, **items}
    normalised = {
        k.lower().strip(): " | ".join(str(i) for i in v) if isinstance(v, list) else v
        for k, v in combined.items()
    }
    return {col: normalised.get(col.lower().strip(), "") for col in COLUMNS}


def create_docs(user_pdf_list):
    df = pd.DataFrame(columns=COLUMNS)
    for pdf_file in user_pdf_list:
        print(f"Processing: {pdf_file}")
        raw_text = get_pdf_text(pdf_file)
        print("Extracting...")
        row = extracted_data(raw_text)
        print(f"Row: {row}")
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        print("Done.")
    return df