import random
from datetime import date, timedelta
from textwrap import dedent


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

    return invoice, date_i, date_d




def render_invoice_text(invoice):

    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}

    vendor_name = invoice.get("vendor_name", "NA")
    invoice_number = invoice.get("invoice_number", "NA")
    invoice_date = date.fromisoformat(invoice["invoice_date"]).strftime("%B %d, %Y")
    due_date = date.fromisoformat(invoice["due_date"]).strftime("%B %d, %Y")

    text = dedent(f"""\
    {vendor_name}

    Invoice Number: {invoice_number}
    Invoice Date: {invoice_date}
    Due Date: {due_date}
    
    subtotal: {symbols[invoice['currency']]}{invoice['subtotal']:,.2f}
    tax: {symbols[invoice['currency']]}{invoice['tax']:,.2f}
    total: {symbols[invoice['currency']]}{invoice['total']:,.2f}
    """)
    return text