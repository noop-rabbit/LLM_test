
# ============================================================
# Standalone eval — fresh session, loads model from checkpoint
# ============================================================

import json
import random
import string
from datetime import date, timedelta
from textwrap import dedent
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
ADAPTER_PATH = "invoice-extractor/checkpoint-246"   # <-- SET THIS (ls invoice-extractor/)

# ============================================================
# Constants (identical to training script)
# ============================================================

FIELDS = ["invoice_number", "invoice_date", "due_date", "subtotal", "tax", "total"]

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

LABELS = {
    "invoice_number": "Invoice Number",
    "invoice_date":   "Invoice Date",
    "due_date":       "Due Date",
    "subtotal":       "Subtotal",
    "tax":            "Tax",
    "total":          "Total",
}

VALID_LABELS = set(LABELS.values()) | {a for v in ALIASES.values() for a in v}
VALID_LOWER = {v.lower() for v in VALID_LABELS}

SYSTEM_PROMPT = """You are an invoice extraction system. Extract invoice fields from the provided text.
Return strict JSON only using the required schema.
Do not include Markdown, explanations, or extra text. Use null for missing values."""

DATE_FMT = "%B %d, %Y"

# ============================================================
# Data generation (identical to training script)
# ============================================================

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


def render_invoice_text(invoice, missing: list = [], unlabeled: list = [], labels: dict = {}):
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "INR": "₹"}

    vendor_name = invoice.get("vendor_name", "NA") if "vendor_name" not in missing else None
    invoice_number = invoice.get("invoice_number", "NA") if "invoice_number" not in missing else ""
    invoice_date = date.fromisoformat(invoice["invoice_date"]).strftime(DATE_FMT) if "invoice_date" not in missing else ""
    due_date = date.fromisoformat(invoice["due_date"]).strftime(DATE_FMT) if "due_date" not in missing else ""
    subtotal = f"{symbols[invoice['currency']]}{invoice['subtotal']:,.2f}" if "subtotal" not in missing else ""
    tax = f"{symbols[invoice['currency']]}{invoice['tax']:,.2f}" if "tax" not in missing else ""
    total = f"{symbols[invoice['currency']]}{invoice['total']:,.2f}" if "total" not in missing else ""

    invoice_number_label = labels.get("invoice_number", "Invoice Number")
    invoice_number_line = f": {invoice_number}" if "invoice_number" in unlabeled else f"{invoice_number_label}: {invoice_number}"

    invoice_date_label = labels.get("invoice_date", "Invoice Date")
    invoice_date_line = f": {invoice_date}" if "invoice_date" in unlabeled else f"{invoice_date_label}: {invoice_date}"

    due_date_label = labels.get("due_date", "Due Date")
    due_date_line = f": {due_date}" if "due_date" in unlabeled else f"{due_date_label}: {due_date}"

    subtotal_label = labels.get("subtotal", "Subtotal")
    subtotal_line = f": {subtotal}" if "subtotal" in unlabeled else f"{subtotal_label}: {subtotal}"

    tax_label = labels.get("tax", "Tax")
    tax_line = f": {tax}" if "tax" in unlabeled else f"{tax_label}: {tax}"

    total_label = labels.get("total", "Total")
    total_line = f": {total}" if "total" in unlabeled else f"{total_label}: {total}"

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


def label_changer(labels=None, num_ch=None):
    labels = labels or []
    num_ch = num_ch or []
    assert len(num_ch) == len(labels)
    result_ch = []
    for i in range(len(labels)):
        text = labels[i]
        k = min(num_ch[i], len(text))
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


def build_target(invoice, hidden: list = []):
    return {
        "vendor_name":    invoice.get("vendor_name")    if "vendor_name"    not in hidden else None,
        "invoice_number": invoice.get("invoice_number") if "invoice_number" not in hidden else None,
        "invoice_date":   date.fromisoformat(invoice["invoice_date"]).strftime(DATE_FMT)
                          if "invoice_date" not in hidden else None,
        "due_date":       date.fromisoformat(invoice["due_date"]).strftime(DATE_FMT)
                          if "due_date" not in hidden else None,
        "currency":       invoice.get("currency")       if "currency"       not in hidden else None,
        "subtotal":       invoice.get("subtotal")       if "subtotal"       not in hidden else None,
        "tax":            invoice.get("tax")            if "tax"            not in hidden else None,
        "total":          invoice.get("total")          if "total"          not in hidden else None,
    }


def make_example(case):
    invoice = generate_invoice_truth()
    missing, unlabeled, labels = [], [], {}
    field = random.choice(FIELDS)
    hidden = []

    if case == "clean":
        pass
    elif case == "missing_value":
        if random.random() < 0.15:
            missing = ["vendor_name"]
            hidden = ["vendor_name"]
        else:
            missing = [field]
            hidden = [field]
    elif case == "label_format_variation":
        labels = {field: random.choice(ALIASES[field])}
    elif case == "accepted_corrupted_label":
        corrupted = label_changer([LABELS[field]], [1])[0]
        while corrupted.lower() in VALID_LOWER:
            corrupted = label_changer([LABELS[field]], [1])[0]
        labels = {field: corrupted}
    elif case == "rejected_corrupted_or_unlabeled":
        if random.random() < 0.3:
            unlabeled = [field]
        else:
            corrupted = label_changer([LABELS[field]], [3])[0]
            while corrupted.lower() in VALID_LOWER:
                corrupted = label_changer([LABELS[field]], [3])[0]
            labels = {field: corrupted}
        hidden = [field]

    text = render_invoice_text(invoice, missing, unlabeled, labels)
    target = build_target(invoice, hidden)
    return {"case": case, "text": text, "target": target}

# ============================================================
# Load base model (4-bit) + trained LoRA adapter
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()
model.config.use_cache = True

# ============================================================
# Fresh eval data — seed 123 (training used 42)
# ============================================================

random.seed(123)
N_PER_CASE = 20    # drop to 20 for a faster first pass

EVAL_CASES = [
    "clean",
    "missing_value",
    "label_format_variation",
    "accepted_corrupted_label",
    "rejected_corrupted_or_unlabeled",
    "hard_misplaced_labels",     # never trained on — generalization test
]


def make_eval_example(case):
    if case == "hard_misplaced_labels":
        ex = make_example("clean")
        lines = [l for l in ex["text"].split("\n") if l.strip()]
        random.shuffle(lines)
        ex["text"] = "\n".join(lines) + "\n"
        ex["case"] = "hard_misplaced_labels"
        return ex
    return make_example(case)


eval_examples = [make_eval_example(c) for c in EVAL_CASES for _ in range(N_PER_CASE)]
print(f"eval set: {len(eval_examples)} examples")

# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def predict(text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,                    # explicit
    ).to(model.device)

    out = model.generate(
        **inputs,                            # unpacks input_ids + attention_mask
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = out[0][inputs["input_ids"].shape[1]:]   # slice off the prompt
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_prediction(raw: str):
    s = raw.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.startswith("json"):
            s = s[4:]
        s = s.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None

# ============================================================
# Scoring
# ============================================================

ALL_FIELDS = ["vendor_name", "invoice_number", "invoice_date", "due_date",
              "currency", "subtotal", "tax", "total"]


def values_equal(pred, truth):
    if isinstance(truth, float) and isinstance(pred, (int, float)):
        return abs(pred - truth) < 1e-6
    return pred == truth


results = defaultdict(lambda: {
    "n": 0, "parse_fail": 0, "all_fields_correct": 0,
    "field_correct": defaultdict(int),
    "null_expected": 0, "null_correct": 0,
    "value_expected": 0, "hallucinated_null": 0,
    "failures": [],
})

for i, ex in enumerate(eval_examples):
    case = ex["case"]
    r = results[case]
    r["n"] += 1

    raw = predict(ex["text"])
    pred = parse_prediction(raw)

    if pred is None:
        r["parse_fail"] += 1
        if len(r["failures"]) < 3:
            r["failures"].append({"text": ex["text"], "raw": raw, "reason": "parse"})
        continue

    field_ok = {f: values_equal(pred.get(f, "MISSING_KEY"), ex["target"][f])
                for f in ALL_FIELDS}
    if all(field_ok.values()):
        r["all_fields_correct"] += 1
    elif len(r["failures"]) < 3:
        wrong = {f: (pred.get(f), ex["target"][f])
                 for f, ok in field_ok.items() if not ok}
        r["failures"].append({"text": ex["text"], "wrong": wrong, "reason": "value"})

    for f, ok in field_ok.items():
        r["field_correct"][f] += ok

    for f in ALL_FIELDS:
        if ex["target"][f] is None:
            r["null_expected"] += 1
            r["null_correct"] += (pred.get(f, "MISSING_KEY") is None)
        else:
            r["value_expected"] += 1
            r["hallucinated_null"] += (pred.get(f, "MISSING_KEY") is None)

    if (i + 1) % 25 == 0:
        print(f"  ... {i + 1}/{len(eval_examples)}")

# ============================================================
# Report
# ============================================================

print("\n" + "=" * 72)
print(f"{'case':<36}{'exact':>8}{'parse✗':>8}{'null-rec':>10}{'halluc-null':>12}")
print("-" * 72)
for case in EVAL_CASES:
    r = results[case]
    exact = r["all_fields_correct"] / r["n"]
    nrec = (r["null_correct"] / r["null_expected"]) if r["null_expected"] else float("nan")
    hnull = r["hallucinated_null"] / r["value_expected"]
    print(f"{case:<36}{exact:>8.1%}{r['parse_fail']:>8}{nrec:>10.1%}{hnull:>12.1%}")
print("=" * 72)
print("exact       = all 8 fields correct")
print("null-rec    = target-None fields correctly output as null")
print("halluc-null = real-valued fields wrongly output as null")


def field_report(case):
    r = results[case]
    n_ok = r["n"] - r["parse_fail"]
    print(f"\n{case} (n={r['n']}, parsed={n_ok})")
    for f in ALL_FIELDS:
        print(f"  {f:<16}{r['field_correct'][f] / max(n_ok, 1):>7.1%}")

# drill in:
# field_report("rejected_corrupted_or_unlabeled")
# results["hard_misplaced_labels"]["failures"]
