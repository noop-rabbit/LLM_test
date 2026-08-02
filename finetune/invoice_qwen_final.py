
import random
import string
import json
from datetime import date, timedelta
from textwrap import dedent
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ============================================================
# Constants
# ============================================================

random.seed(42)
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

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

case_dictionary = {
    "clean": 750,
    "rejected_corrupted_or_unlabeled": 200,
    "accepted_corrupted_label": 150,
    "missing_value": 150,
    "label_format_variation": 200,
}

SYSTEM_PROMPT = """You are an invoice extraction system. Extract invoice fields from the provided text.
Return strict JSON only using the required schema.
Do not include Markdown, explanations, or extra text. Use null for missing values."""

DATE_FMT = "%B %d, %Y"   # shared by renderer AND build_target

# ============================================================
# Truth generator
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

# ============================================================
# Renderer
# ============================================================

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

# ============================================================
# Label corruptors
# (label_formatter/case_converter currently unused by the planner —
#  kept for the future case-variation transform)
# ============================================================

def label_formatter(labels=None):
    labels = labels or []
    result_lab = []
    for i in range(len(labels)):
        text = labels[i]
        valid_positions = [l for l, ch in enumerate(text) if ch.isalpha()]
        if not valid_positions:
            result_lab.append(text)
            continue
        k = random.randint(1, len(valid_positions))
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

# ============================================================
# Target builder
# ============================================================

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

# ============================================================
# Case planner
# ============================================================

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


def to_chat_example(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": example["text"]},
        ],
        "completion": [
            {"role": "assistant", "content": json.dumps(example["target"], indent=2)},
        ],
    }

# ============================================================
# Build dataset + verify + write JSONL
# ============================================================

train_dataset, val_dataset = [], []
for key, value in case_dictionary.items():
    for itr in range(value):
        if itr < int(0.9 * value):
            train_dataset.append(make_example(key))
        else:
            val_dataset.append(make_example(key))

random.shuffle(train_dataset)
random.shuffle(val_dataset)

# --- verify counts (expected: 675/180/135/135/180 and 75/20/15/15/20) ---
print("train:", Counter(e["case"] for e in train_dataset))
print("val:  ", Counter(e["case"] for e in val_dataset))

train_chat = [to_chat_example(e) for e in train_dataset]
val_chat = [to_chat_example(e) for e in val_dataset]

with open("invoice_train.jsonl", "w", encoding="utf-8") as f:
    for ex in train_chat:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

with open("invoice_val.jsonl", "w", encoding="utf-8") as f:
    for ex in val_chat:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

ds = load_dataset("json", data_files={"train": "invoice_train.jsonl",
                                      "validation": "invoice_val.jsonl"})
print(ds)                                # 1305 / 145
print(ds["train"][0]["prompt"])          # FIX: was ["messages"] — that column no longer exists
print(ds["train"][0]["completion"])

# ============================================================
# Tokenizer + max-length check
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

lens = [len(tokenizer.apply_chat_template(ex["prompt"] + ex["completion"],
                                          tokenize=True, return_dict=False))   # ✓ counts tokens
        for ex in train_chat]
max_len = max(lens)
print(f"token lengths: max={max_len}, avg={sum(lens)/len(lens):.0f}")
assert max_len < 512, f"increase max_length: longest example is {max_len} tokens"

# ============================================================
# Model (QLoRA, RTX 3060 12 GB)
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
model.config.use_cache = False

print("VRAM after load:", torch.cuda.max_memory_allocated() / (1024**3), "GB")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# Trainer
# ============================================================

sft_config = SFTConfig(
    output_dir="invoice-extractor",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,      # effective batch 16
    per_device_eval_batch_size=2,
    learning_rate=2e-4,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="epoch",
    bf16=True,
    max_length=512,                      # measured above; halves activation memory vs 1024
    completion_only_loss=True,
    optim="paged_adamw_8bit",            # QLoRA-standard: paged 8-bit optimizer
    report_to="none",                    # no wandb prompt
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=ds["train"],
    eval_dataset=ds["validation"],
    processing_class=tokenizer,
)

# ============================================================
# Pre-flight: verify prompt masking — GATES training
# ============================================================

batch = next(iter(trainer.get_train_dataloader()))
labels = batch["labels"][0]
print("labels[:50]:", labels[:50])
print("unmasked tokens:", (labels != -100).sum().item())

assert (labels[:10] == -100).all(), "MASKING FAILED: prompt tokens are in the loss!"
assert (labels != -100).sum() > 0, "MASKING FAILED: completion is fully masked!"
print("mask check passed ✓")

# ============================================================
# Train
# ============================================================

trainer.train()
