from fastapi import FastAPI, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import pandas as pd
import requests
import json
import re
import os

# ---- CONFIGURATION ----
EXCEL_FILE = os.getenv('EXCEL_FILE', 'Data_Dictionary_New_4.xlsx')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/generate')
MODEL_NAME = os.getenv('MODEL_NAME', 'codellama:7b-instruct')
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '256'))

# ---- LOAD SCHEMA ON STARTUP ----
def load_schema_from_excel(
    file_path,
    schema_col='Schema Name',
    table_col='Table Name',
    col_col='Data Element /  Data Member Name / Configurable Item', 
    dtype_col='Element Format',
    desc_col='Description'
):
    df = pd.read_excel(file_path)
    df.columns = [str(c).strip() for c in df.columns]
    required = [table_col, col_col]
    for req in required:
        if req not in df.columns:
            raise ValueError(f"Missing required column: {req}")

    schema = {}
    for _, row in df.iterrows():
        schema_name = str(row.get(schema_col, "")).strip()
        table_name = str(row[table_col]).strip()
        if not table_name:
            continue
        full_table = f"{schema_name}.{table_name}" if schema_name else table_name
        column = str(row[col_col]).strip()
        if not column:
            continue
        dtype = str(row.get(dtype_col, "")).strip()
        desc = str(row.get(desc_col, "")).strip()
        line = f"- {column}" + (f": {dtype}" if dtype else "") + (f" -- {desc}" if desc else "")
        schema.setdefault(full_table, []).append(line)
    return schema

def format_schema_prompt(schema_dict, selected_tables=None):
    lines = []
    for table, fields in schema_dict.items():
        if selected_tables and table not in selected_tables:
            continue
        if not fields:
            continue
        lines.append(f"Table: {table}")
        lines.extend(fields)
        lines.append("")
    return "\n".join(lines).strip()

def filter_tables_by_column(schema_dict, column_name):
    if not column_name:
        return {}
    col_upper = column_name.strip().upper()
    result = {}
    for table, columns in schema_dict.items():
        for col_line in columns:
            col = col_line.lstrip('-').split(':', 1)[0].strip().upper()
            if col == col_upper:
                result[table] = columns
                break
    return result

def extract_column_name_from_user_prompt(user_prompt):
    return re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', user_prompt)

def build_prompt(data_dictionary, user_prompt):
    return f"""You are a SQL code assistant working with this database schema:

{data_dictionary}

CRITICAL INSTRUCTIONS:
- Write all queries in Oracle SQL dialect, using Oracle-specific syntax.
- Use ONLY exact column names and schema-qualified table names (SCHEMA.TABLE) exactly as shown above.
- When generating queries, if a column name appears in more than one table used in the query (overlapping column names), you must always qualify every use of that column (in SELECT, WHERE, ON, GROUP BY, ORDER BY, etc.) with the table name or table alias—even if you think it is unambiguous in context.
- Never use ambiguous column names in any part of the query.
- For every column in the SELECT clause, always use a unique output alias if the column exists in more than one source table to clearly identify which table it came from.
- When joining tables, always define and use table aliases, and qualify every referenced column accordingly.
- If you are unsure which table a column should come from (for columns with the same name in multiple tables), use join relationships to select the most logical source table based on schema, and/or output with a comment clarifying the choice.
- If no join or relationship makes it clear which source table to use for a column common to multiple tables, prefer the table listed first in FROM or document the ambiguity if unsure.
- Confirm every column you use exists in the specified table. Never guess.
- Use schema-qualified table names (SCHEMA.TABLE format).
- For Oracle SQL, limit result sets using FETCH FIRST N ROWS ONLY at the end.
- Payment method information is only to be taken from tables whose names include "PAYMENTMETHOD".
- Only 'PAYMENTMETHOD' table is from a schema named 'PRODUCTMASTER_COMMONS'. Eg: PRODUCTMASTER_PAYPRO.BAPAYMENTMETHOD.
- Transaction data is only to be retrieved from tables whose names include "BATCH" or "TRANSACTION".

APPROACH TO AMBIGUITY:
1. Identify all columns needed, and check if each exists in more than one relevant table.
2. For all columns in any SELECT, WHERE, JOIN, ON, GROUP BY, or ORDER BY clause, always qualify with the corresponding table or alias.
3. Always use unique aliases in SELECT for fields present in multiple tables.
4. If any ambiguity remains (same-named columns in multiple joined tables and it’s unclear which to use), either:
    - Select both columns and clearly alias each, or
    - Choose one based on join relationships, and in code comments clarify which table it’s from.

EXAMPLES:
Ambiguity-resilient SELECT:

sql
SELECT
    b.ID AS batch_id,
    b.CORPORATEREFNO,
    pm.PAYMENTMETHODNAME AS payment_method,
    b.ENTRYDATE
FROM
    PRODUCTMASTER_PAYPRO.BTBATCHMST b
    LEFT JOIN PRODUCTMASTER_PAYPRO.BAPAYMENTMETHOD pm ON b.PAYMENTMETHODID = pm.ID

(Note how ENTRYDATE is qualified from both tables.)

When in doubt, always clearly qualify columns to eliminate all ambiguity.

SCHEMA ANALYSIS CHECKLIST:
- Check schema for every table and column used.
- Use table alias and/or schema.table format for all columns when multiple tables are in the query or when columns share names across tables.
- Add unique output aliases for columns from different tables if they share the same name.

Example:
Schema Appropriate Select Query:

Q: Write a query to get every row from REQUESTBATCHID
A: SELECT REQUESTBATCHID FROM PRODUCTMASTER_PAYPRO.BTH2HREPORT;

Question: write a query to {user_prompt}

Answer with ONLY the SQL query:
"""

def generate_sql(prompt):
    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "options": {
            "temperature": 0.2,
            "num_predict": MAX_TOKENS
        }
    }
    response = requests.post(OLLAMA_URL, json=data, stream=True)
    response.raise_for_status()
    sql_parts = []
    for line in response.iter_lines():
        if line:
            obj = json.loads(line.decode("utf-8"))
            sql_parts.append(obj.get("response", ""))
    return "".join(sql_parts).strip()

# Load schema at startup once
schema_dict = load_schema_from_excel(EXCEL_FILE)

# ---- FASTAPI SETUP ----
app = FastAPI(title="Codellama SQL Generator API")

class SQLRequest(BaseModel):
    user_prompt: str

@app.post("/generate-sql-v1")
def generate_sql_endpoint(req: SQLRequest):
    candidate_cols = extract_column_name_from_user_prompt(req.user_prompt)
    filtered_schema = schema_dict
    if candidate_cols:
        for col in candidate_cols:
            filtered = filter_tables_by_column(schema_dict, col)
            if filtered:
                filtered_schema = filtered
                break

    data_dictionary = format_schema_prompt(filtered_schema)
    prompt = build_prompt(data_dictionary, req.user_prompt)
    sql_query = generate_sql(prompt)

    # 🔹 Clean the SQL to remove newlines / tabs before returning
    # clean_sql = sql_query.replace("\n", " ").replace("\r", " ").strip()
    # cleaned_sql = clean_sql.replace("\t", "").strip()

    return PlainTextResponse(sql_query)
