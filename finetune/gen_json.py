import json

# Standard database schema templates to generate 200 distinct conversations
schemas = [
    ("users", "id, name, signup_year, country, age", "country", "'USA'", "age > 21", "signup_year > 2020"),
    ("employees", "id, name, department, salary, hire_year", "department", "'Engineering'", "salary > 80000", "hire_year = 2022"),
    ("products", "id, name, category, price, stock", "category", "'Electronics'", "price < 500", "stock > 0"),
    ("orders", "order_id, customer_id, order_date, amount, status", "status", "'Completed'", "amount > 100", "YEAR(order_date) = 2024"),
    ("students", "id, name, major, gpa, graduation_year", "major", "'Computer Science'", "gpa > 3.5", "graduation_year = 2025"),
    ("movies", "id, title, genre, rating, release_year", "genre", "'Action'", "rating > 8.0", "release_year > 2015"),
    ("sales", "id, region, year, revenue, units_sold", "region", "'North'", "year = 2023", "units_sold > 500"),
    ("books", "id, title, author, genre, price", "genre", "'Fiction'", "price < 20", "author = 'King'"),
    ("patients", "id, name, age, gender, admitted_year", "gender", "'F'", "age > 65", "admitted_year = 2023"),
    ("flights", "flight_id, airline, origin, destination, price", "airline", "'Delta'", "destination = 'JFK'", "price < 400")
]

dataset = []
index = 1

for i in range(20):  # 20 variations of 10 schemas = 200 samples
    for table, cols, f1_col, f1_val, f2_cond, f3_cond in schemas:
        tbl_name = f"{table}_{i+1}" if i > 0 else table
        
        data_point = {
            "schema": f"{tbl_name}({cols})",
            "turns": [
                {
                    "q": f"Show all records from {tbl_name}.",
                    "a": f"SELECT * FROM {tbl_name};"
                },
                {
                    "q": f"Only where {f1_col} is {f1_val}.",
                    "a": f"SELECT * FROM {tbl_name} WHERE {f1_col} = {f1_val};"
                },
                {
                    "q": f"Filter further by {f2_cond}.",
                    "a": f"SELECT * FROM {tbl_name} WHERE {f1_col} = {f1_val} AND {f2_cond};"
                },
                {
                    "q": f"And also {f3_cond}.",
                    "a": f"SELECT * FROM {tbl_name} WHERE {f1_col} = {f1_val} AND {f2_cond} AND {f3_cond};"
                }
            ]
        }
        dataset.append(data_point)

with open("dataset_200.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Successfully created dataset_200.json with 200 entries!")