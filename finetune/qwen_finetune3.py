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
import string

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

def render_invoice_text(invoice, missing: list = [], unlabeled: list = [], labels: dict = {}):
  
  
  symbols = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}

  vendor_name = invoice.get("vendor_name", "NA") if "vendor_name" not in missing else None
  invoice_number = invoice.get("invoice_number", "NA") if "invoice_number" not in missing else ""
  invoice_date = date.fromisoformat(invoice["invoice_date"]).strftime("%B %d, %Y") if "invoice_date" not in missing else ""
  due_date = date.fromisoformat(invoice["due_date"]).strftime("%B %d, %Y") if "due_date" not in missing else ""
  subtotal = f"{symbols[invoice['currency']]}{invoice['subtotal']:,.2f}" if "subtotal" not in missing else ""
  tax = f"{symbols[invoice['currency']]}{invoice['tax']:,.2f}" if "tax" not in missing else ""
  total= f"{symbols[invoice['currency']]}{invoice['total']:,.2f}" if "total" not in missing else ""

  invoice_number_label = labels.get("invoice_number", "Invoice Number")
  invoice_number_line= f": {invoice_number}" if "invoice_number" in unlabeled else f"{invoice_number_label}: {invoice_number}"

  invoice_date_label = labels.get("invoice_date", "Invoice Date")
  invoice_date_line= f": {invoice_date}" if "invoice_date" in unlabeled else f"{invoice_date_label}: {invoice_date}"

  due_date_label = labels.get("due_date", "Due Date")
  due_date_line= f": {due_date}" if "due_date" in unlabeled else f"{due_date_label}: {due_date}"

  subtotal_label = labels.get("subtotal", "Subtotal")
  subtotal_line= f": {subtotal}" if "subtotal" in unlabeled else f"{subtotal_label}: {subtotal}"

  tax_label = labels.get("tax", "Tax")
  tax_line= f": {tax}" if "tax" in unlabeled else f"{tax_label}: {tax}"

  total_label = labels.get("total", "Total")
  total_line= f": {total}" if "total" in unlabeled else f"{total_label}: {total}"
  
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
print(render_invoice_text(invoice, unlabeled=["subtotal"], labels={"tax": "x7Q"}))
print(render_invoice_text(invoice, missing=["total"], labels={"invoice_number": "Ref No"})) 

def label_formatter(labels=None):
   labels = labels or []
   
   result_lab = []
   for i in range(len(labels)):
  
    text = labels[i]
    valid_positions = [l for l, ch in enumerate(text) if ch.isalpha()]

    if not valid_positions:
      result_lab.append(text)
      continue

    k = random.randint(1, len(valid_positions))              ## k = number of letter change
    choices = random.sample(valid_positions, k)  

    for j in choices:      
      text = case_converter(labels_c=text, let_num=j)

    result_lab.append(text)

   return result_lab
    


def case_converter(labels_c: str = "", let_num: int = 0) -> str:
  result = ""

  if not labels_c or not (0 <= let_num < len(labels_c)):
       return labels_c

  for i in range(len(labels_c)):
    if i == let_num and labels_c[i].islower():
      result += labels_c[i].upper()
    elif i == let_num and labels_c[i].isupper():
      result += labels_c[i].lower()
    else:
      result += labels_c[i]

  return result



def label_changer(labels=None, num_ch=None):

   labels = labels or []
   num_ch = num_ch or []
   
   assert len(num_ch) == len(labels)
   result_ch = []

   for i in range(len(labels)):
    text = labels[i]

    k = min(num_ch[i], len(text))     ## k = number of letter change

    choices = random.sample(range(len(text)), k)
    for j in choices:
      text = case_dest(labels_c=text, let_num=j)
    result_ch.append(text)
   return result_ch
    


def case_dest(labels_c: str = "", let_num: int = 0) -> str:
  result_de = ""
  characters = string.ascii_letters + string.digits

  if not labels_c or not (0 <= let_num < len(labels_c)):
       return labels_c

  for i in range(len(labels_c)):
    if i == let_num:
      result_de += random.choice([c for c in characters if c != labels_c[i]])
    else:
      result_de += labels_c[i]

  return result_de


# print(label_formatter(["Tax", "Invoice Number"]))
# print(label_changer(["Tax", "Invoice Number"], [1, 3]))

ALIASES = {
    "invoice_number": ["Invoice No", "Inv No", "Reference", "Ref No",
                       "Invoice No.", "Inv No.", "Ref No."],
    "invoice_date":   ["Inv Date", "Date of Invoice", "Bill Date",
                       "Issue Date", "Date of Issue"],
    "due_date":       ["Payment Due", "Due On", "Pay By", "Last Date"],
    "subtotal":       ["Sub Total", "Sub-Total", "Net Amount", "Amount Before Tax"],
    "tax":            ["Tax Amt", "VAT", "GST", "Sales Tax"],
    "total":          ["Total Due", "Amount Due", "Grand Total", "Balance Due",
                       "Amount After Tax"],
}