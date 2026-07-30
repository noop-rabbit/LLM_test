import random
import json
import subprocess
from datetime import date, timedelta
from textwrap import dedent

# Different seed than training (42) → fresh, unseen invoices
random.seed(777)

def generate_invoice_truth():
    mylist1 = ["Apple", "Banana", "Cherry", "Milk", "Mango", "Sugar"]
    mylist2 = ["Gov.", "Republic", "Ltd.", "Enterprise", "Corp.", "Inc."]
    mylist3 = ["USD", "GBP", "INR", "EUR"]
    days = [15, 30, 45]

    amount = round(random.uniform(1, 10000), 2)
    rate = random.randint(5, 20) / 100
    tax = round(amount * rate, 2)
    total = round(amount + tax, 2)

    date_i = date(2024, 3, 8) + timedelta(days=random.randint(1, 1000))
    date_d = date_i + timedelta(days=random.choice(days))

    inv_num = random.choice(["INV", "REF", "BILL"]) + "-" + str(random.randint(1000, 99999))

    return {
        "vendor_name": random.choice(mylist1) + " " + random.choice(mylist2),
        "invoice_number": inv_num,
        "invoice_date": date_i.isoformat(),
        "due_date": date_d.isoformat(),
        "currency": random.choice(mylist3),
        "subtotal": amount,
        "tax": tax,
        "total": total,
    }

def render_invoice_text(invoice):
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}
    invoice_date = date.fromisoformat(invoice["invoice_date"]).strftime("%B %d, %Y")
    due_date = date.fromisoformat(invoice["due_date"]).strftime("%B %d, %Y")

    return dedent(f"""\
    {invoice['vendor_name']}

    Invoice Number: {invoice['invoice_number']}
    Invoice Date: {invoice_date}
    Due Date: {due_date}

    Subtotal: {symbols[invoice['currency']]}{invoice['subtotal']:,.2f}
    Tax: {symbols[invoice['currency']]}{invoice['tax']:,.2f}
    Total: {symbols[invoice['currency']]}{invoice['total']:,.2f}
    """)

# ---------------- Run the batch test ----------------

N = 20
valid_json = 0
correct_values = 0
required_keys = {"vendor_name", "invoice_number", "invoice_date", "due_date",
                 "currency", "subtotal", "tax", "total"}

for i in range(N):
    truth = generate_invoice_truth()
    text = render_invoice_text(truth)

    r = subprocess.run(
        ["ollama", "run", "qwen-invoice", text],
        capture_output=True, text=True, timeout=120,
    )
    output = r.stdout.strip()

    # 1. Does it parse?
    try:
        parsed = json.loads(output)
        valid_json += 1
    except json.JSONDecodeError:
        print(f"[{i}] INVALID JSON:\n{output[:300]}\n")
        continue

    # 2. Right keys + right values?
    if set(parsed.keys()) == required_keys and parsed == truth:
        correct_values += 1
    else:
        diffs = {k: (parsed.get(k), truth[k]) for k in required_keys
                 if parsed.get(k) != truth[k]}
        print(f"[{i}] JSON ok, value mismatch: {diffs}")

print(f"\nValid JSON:      {valid_json}/{N}")
print(f"Fully correct:   {correct_values}/{N}")