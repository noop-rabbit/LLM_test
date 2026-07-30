import random
from datetime import date, timedelta
from textwrap import dedent
import torch
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import torch.optim as optim
import bitsandbytes as bnb
import json
from datasets import load_dataset

def generate_invoice_truth():

  mylist1 = ["Apple", "Banana", "Cherry", "Milk", "Mango", "Sugar"]
  mylist2 = ["Gov.", "Republic", "Ltd.", "Enterprise", "Corp.", "Inc."]
  mylist3 = ["USD", "GBP", "INR", "EUR"]
  days = [15, 30, 45]

  amount = round(random.uniform(1, 10000), 2)
  rate = random.randint(5, 20) / 100
  tax = round(amount * rate, 2)
  total = round(amount + tax, 2)

  date_i = date(2024, 3, 8) + timedelta(days=random.randint(1,1000))
  date_d = date_i + timedelta(days=random.choice(days))

  inv_num = random.choice(["INV", "REF", "BILL"]) + "-" + str(random.randint(1000, 99999))

  invoice = {
      "vendor_name" : random.choice(mylist1)+ " " + random.choice(mylist2),
      "invoice_number" :  inv_num,
      "invoice_date" : date_i.isoformat(),
      "due_date" : date_d.isoformat(),
      "currency" : random.choice(mylist3),
      "subtotal" : amount,
      "tax" : tax,
      "total" : total,
  }

  return invoice

def render_invoice_text(invoice, missing: list = [], unlabeled: list = []):
  
  
  symbols = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}

  vendor_name = invoice.get("vendor_name", "NA") if "vendor_name" not in missing else None
  invoice_number = invoice.get("invoice_number", "NA") if "invoice_number" not in missing else ""
  invoice_date = date.fromisoformat(invoice["invoice_date"]).strftime("%B %d, %Y") if "invoice_date" not in missing else ""
  due_date = date.fromisoformat(invoice["due_date"]).strftime("%B %d, %Y") if "due_date" not in missing else ""
  subtotal = f"{symbols[invoice['currency']]}{invoice['subtotal']:,.2f}" if "subtotal" not in missing else ""
  tax = f"{symbols[invoice['currency']]}{invoice['tax']:,.2f}" if "tax" not in missing else ""
  total= f"{symbols[invoice['currency']]}{invoice['total']:,.2f}" if "total" not in missing else ""

  invoice_number_line= f": {invoice_number}" if "invoice_number" in unlabeled else f"Invoice Number: {invoice_number}"
  invoice_date_line= f": {invoice_date}" if "invoice_date" in unlabeled else f"Invoice Date: {invoice_date}"
  due_date_line= f": {due_date}" if "due_date" in unlabeled else f"Due Date: {due_date}"
  subtotal_line= f": {subtotal}" if "subtotal" in unlabeled else f"Subtotal: {subtotal}"
  tax_line= f": {tax}" if "tax" in unlabeled else f"Tax: {tax}"
  total_line= f": {total}" if "total" in unlabeled else f"Total: {total}"
  
  header = f"{vendor_name}\n\n" if vendor_name else ""

  text = header + dedent(f"""\
  {invoice_number_line}
  {invoice_date_line}
  {due_date_line}
  
  {subtotal_line}
  {tax_line}
  {total_line}
  """)

  return text

invoice = generate_invoice_truth()
print(render_invoice_text(invoice)) 