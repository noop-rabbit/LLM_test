import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_DIR = "./qlora_output/final_adapter"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Tokenizer — note: from the ADAPTER dir, not the hub
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)

model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()
model.config.use_cache = True



# messages = [{"role": "user",
# #              "content": "books(id, title, author, published_year)\n\nHow many books?"}]

# messages = [
#     {"role": "user", "content": "books(id, title, author, published_year)\n\nHow many books?"},
#     {"role": "assistant", "content": "SELECT COUNT(*) FROM books;"},
#     {"role": "user", "content": "Only those published after 2000."},
# ]

messages = [
    {"role": "user", "content": "books(id, title, author, published_year, genre, price)\nauthors(id, name, country, birth_year)\npublishers(id, name, country)\nreviews(id, book_id, user_id, rating, comment, review_date)\nusers(id, name, email, signup_date, country)\norders(id, user_id, book_id, quantity, order_date, total_price)\n\nHow many books do we have in total?"},
    {"role": "assistant", "content": "SELECT COUNT(*) FROM books;"},
    {"role": "user", "content": "Only those published after 2000."},
    {"role": "assistant", "content": "SELECT COUNT(*) FROM books WHERE published_year > 2000;"},
    {"role": "user", "content": "Now break that down by genre."},
    {"role": "assistant", "content": "SELECT genre, COUNT(*) AS book_count\nFROM books\nWHERE published_year > 2000\nGROUP BY genre\nORDER BY book_count DESC;"},
    {"role": "user", "content": "Which genre has the highest average price?"},
    {"role": "assistant", "content": "SELECT genre, AVG(price) AS avg_price\nFROM books\nGROUP BY genre\nORDER BY avg_price DESC\nLIMIT 1;"},
    {"role": "user", "content": "Give me the top 5 authors by number of books published."},
    {"role": "assistant", "content": "SELECT author, COUNT(*) AS num_books\nFROM books\nGROUP BY author\nORDER BY num_books DESC\nLIMIT 5;"},
    {"role": "user", "content": "I actually want author details from the authors table, joined properly, and include their country."},
    {"role": "assistant", "content": "SELECT a.name, a.country, COUNT(b.id) AS num_books\nFROM authors a\nJOIN books b ON b.author = a.name\nGROUP BY a.name, a.country\nORDER BY num_books DESC\nLIMIT 5;"},
    {"role": "user", "content": "Now show me the average rating for each book, along with title and author."},
    {"role": "assistant", "content": "SELECT b.title, b.author, AVG(r.rating) AS avg_rating\nFROM books b\nJOIN reviews r ON r.book_id = b.id\nGROUP BY b.id, b.title, b.author\nORDER BY avg_rating DESC;"},
    {"role": "user", "content": "Only include books with at least 10 reviews."},
    {"role": "assistant", "content": "SELECT b.title, b.author, AVG(r.rating) AS avg_rating, COUNT(r.id) AS review_count\nFROM books b\nJOIN reviews r ON r.book_id = b.id\nGROUP BY b.id, b.title, b.author\nHAVING COUNT(r.id) >= 10\nORDER BY avg_rating DESC;"},
    {"role": "user", "content": "Which users have spent the most money overall, across all their orders?"},
    {"role": "assistant", "content": "SELECT u.name, SUM(o.total_price) AS total_spent\nFROM users u\nJOIN orders o ON o.user_id = u.id\nGROUP BY u.id, u.name\nORDER BY total_spent DESC;"},
    {"role": "user", "content": "Limit that to users from the United States, top 10 only."},
    {"role": "assistant", "content": "SELECT u.name, SUM(o.total_price) AS total_spent\nFROM users u\nJOIN orders o ON o.user_id = u.id\nWHERE u.country = 'United States'\nGROUP BY u.id, u.name\nORDER BY total_spent DESC\nLIMIT 10;"},
    {"role": "user", "content": "For each publisher, show total revenue from orders of their books, using the publishers table."},
    {"role": "assistant", "content": "SELECT p.name, SUM(o.total_price) AS total_revenue\nFROM publishers p\nJOIN books b ON b.id = o.book_id\nJOIN orders o ON o.book_id = b.id\nGROUP BY p.id, p.name\nORDER BY total_revenue DESC;"},
    {"role": "user", "content": "That join looks off, books doesn't have a publisher_id in the schema I gave, so just ignore publishers for now. Instead, find the month with the highest total order revenue."},
    {"role": "assistant", "content": "SELECT strftime('%Y-%m', order_date) AS month, SUM(total_price) AS monthly_revenue\nFROM orders\nGROUP BY month\nORDER BY monthly_revenue DESC\nLIMIT 1;"},
    {"role": "user", "content": "Show me a running total of revenue by month, ordered chronologically."},
    {"role": "assistant", "content": "SELECT strftime('%Y-%m', order_date) AS month,\n       SUM(total_price) AS monthly_revenue,\n       SUM(SUM(total_price)) OVER (ORDER BY strftime('%Y-%m', order_date)) AS running_total\nFROM orders\nGROUP BY month\nORDER BY month;"},
    {"role": "user", "content": "Now find users who have never left a review despite having ordered at least one book."},
    
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,          # be explicit, don't rely on the default
).to("cuda")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=100, do_sample=False)

prompt_len = inputs["input_ids"].shape[1]
print(tokenizer.decode(out[0][prompt_len:]))