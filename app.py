import os
import sqlite3
import random
import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =========================
#  Basic configuration
# =========================

load_dotenv()

# Hugging Face token (NOT OpenAI anymore)
HF_API_KEY = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Default LLM model for chat + tools
# :hf-inference -> HF Inference provider orqali ishlashini bildiradi
DEFAULT_LLM_MODEL = os.getenv(
    "HF_LLM_MODEL_ID",
    "deepseek-ai/DeepSeek-R1:fastest",
)

# Hugging Face Router uchun to‘g‘ri endpoint
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_API_KEY,
)


BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "sales_data.db"
TICKETS_PATH = BASE_DIR / "support_tickets.csv"


# =========================
#  Utility: logging
# =========================

def init_session_state():
    """Initialize Streamlit session state variables for chat and logs."""
    if "chat_history" not in st.session_state:
        # This is only for UI display
        st.session_state.chat_history = []

    if "llm_messages" not in st.session_state:
        # This is internal history we send to the LLM (system+user+assistant+tool)
        st.session_state.llm_messages = []

    if "logs" not in st.session_state:
        st.session_state.logs = []

    if "ticket_counter" not in st.session_state:
        st.session_state.ticket_counter = 1


def log_event(message: str):
    """Append timestamped log message to the session logs and show in console."""
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {message}"
    st.session_state.logs.append(entry)
    # For developer debugging in console as well
    print(entry)


# =========================
#  Database creation & seeding
# =========================

def init_database():
    """Create SQLite database and seed it with at least 500 rows if empty."""
    created = not DB_PATH.exists()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_date TEXT,
            region TEXT,
            product TEXT,
            customer_segment TEXT,
            quantity INTEGER,
            unit_price REAL,
            total_amount REAL
        )
        """
    )

    cursor.execute("SELECT COUNT(*) FROM sales")
    row_count = cursor.fetchone()[0]

    if row_count < 500:
        log_event("Seeding database with synthetic sales data (>=500 rows).")
        seed_sales_data(cursor, min_rows=600)
        conn.commit()
    else:
        if created:
            log_event("Database created and already contains data.")
        else:
            log_event("Database already exists with enough data.")

    conn.close()


def seed_sales_data(cursor, min_rows=600):
    """Generate synthetic sales data for the demo database."""
    regions = ["North", "South", "East", "West", "Central"]
    products = ["Laptop", "Smartphone", "Tablet", "Headphones", "Monitor"]
    segments = ["Retail", "Corporate", "Online", "Reseller"]

    today = datetime.today()
    start_date = today - timedelta(days=365)

    for _ in range(min_rows):
        order_date = start_date + timedelta(days=random.randint(0, 365))
        region = random.choice(regions)
        product = random.choice(products)
        segment = random.choice(segments)
        quantity = random.randint(1, 20)
        unit_price = round(random.uniform(50, 2000), 2)
        total_amount = round(quantity * unit_price, 2)

        cursor.execute(
            """
            INSERT INTO sales (
                order_date, region, product, customer_segment,
                quantity, unit_price, total_amount
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_date.strftime("%Y-%m-%d"),
                region,
                product,
                segment,
                quantity,
                unit_price,
                total_amount,
            ),
        )


# =========================
#  Data access helpers (tools)
# =========================

def safe_run_sql_query(query: str) -> dict:
    """
    Safely execute a SELECT query on the SQLite database.

    - Blocks any dangerous operations (DELETE, UPDATE, DROP, etc.).
    - Limits large results.
    - Returns a dict with columns, rows, and row_count.
    """
    if not query:
        raise ValueError("Query is empty.")

    lowered = query.strip().lower()

    dangerous_keywords = [
        "delete", "update", "insert", "drop", "alter",
        "truncate", "create", "replace"
    ]
    if any(k in lowered for k in dangerous_keywords):
        raise ValueError("Dangerous SQL operation detected. Only SELECT queries are allowed.")

    if not lowered.startswith("select"):
        raise ValueError("Only SELECT queries are allowed in this app.")

    # Add LIMIT if user forgot (to protect from huge outputs)
    if "limit" not in lowered:
        query = query.strip().rstrip(";") + " LIMIT 100;"

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    conn.close()

    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
    }


def get_table_overview() -> dict:
    """
    Return high-level information about the 'sales' table:
    - row_count
    - distinct regions
    - distinct products
    - date range
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM sales")
    row_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT region) FROM sales")
    region_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT product) FROM sales")
    product_count = cursor.fetchone()[0]

    cursor.execute("SELECT MIN(order_date), MAX(order_date) FROM sales")
    min_date, max_date = cursor.fetchone()

    conn.close()

    return {
        "row_count": row_count,
        "region_count": region_count,
        "product_count": product_count,
        "min_date": min_date,
        "max_date": max_date,
    }


def create_support_ticket(description: str, user_email: str | None = None,
                          severity: str | None = "medium") -> dict:
    """
    Simulate support ticket creation.

    For Capstone Project 1 we treat this as creating a ticket in
    some external system (GitHub / Trello / Jira equivalent).
    Here we simply append to a local CSV file.
    """
    if not description:
        raise ValueError("Support ticket description cannot be empty.")

    TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    ticket_id = st.session_state.ticket_counter
    st.session_state.ticket_counter += 1

    created_at = datetime.now().isoformat(timespec="seconds")

    # Append to CSV
    import csv
    file_exists = TICKETS_PATH.exists()

    with open(TICKETS_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ticket_id", "created_at", "severity",
                             "user_email", "description", "status"])
        writer.writerow([ticket_id, created_at, severity, user_email, description, "open"])

    return {
        "ticket_id": ticket_id,
        "created_at": created_at,
        "severity": severity,
        "user_email": user_email,
        "status": "open",
        "message": "Support ticket created successfully.",
    }


# =========================
#  LLM tools schema (for function calling)
# =========================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql_query",
            "description": "Run a safe SQL SELECT query on the sales database and return rows.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A SQL SELECT query that reads from the 'sales' table only."
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_table_overview",
            "description": "Get a high-level overview of the sales table (row count, date range, dimensions).",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": (
                "Create a support ticket for the human team when the user "
                "needs help beyond the agent’s capabilities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Summary of the user issue that should be sent to support."
                    },
                    "user_email": {
                        "type": "string",
                        "description": "Optional contact email of the user.",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "How urgent the issue is.",
                    },
                },
                "required": ["description"],
            },
        },
    },
]

SYSTEM_PROMPT = """
You are a Data Insights Assistant for the "Data Insights App – Capstone Project 1".

You work with a SQLite database called 'sales' that contains fields:
- order_date (YYYY-MM-DD)
- region
- product
- customer_segment
- quantity
- unit_price
- total_amount

Your main tasks:
1. Help the user analyze the sales data.
2. Use tools when needed:
   - run_sql_query: to fetch specific rows or aggregated data.
   - get_table_overview: to understand the global structure.
   - create_support_ticket: if the user needs human support or the issue is outside your scope.

VERY IMPORTANT SAFETY RULE:
- You MUST NOT execute any SQL that modifies data (no DELETE, UPDATE, INSERT, DROP, ALTER, TRUNCATE, CREATE, REPLACE).
- Only safe SELECT queries are allowed. If the user asks for dangerous operations, explain that it is blocked by safety policy.

Tool usage policy:
- When the user asks about data (numbers, time periods, totals, etc.), prefer using run_sql_query or get_table_overview.
- If the question is more conceptual (e.g., explaining KPIs) and no exact numbers are needed, you may answer without tools.
- If the user seems frustrated, repeatedly mentions issues, or explicitly asks for human support,
  you SHOULD call create_support_ticket and then explain that a ticket has been created.

Response style:
- Answer briefly and clearly in English.
- When you show data results, summarize key points instead of dumping everything.
""".strip()


# =========================
#  LLM + tools orchestration
# =========================

def call_agent_with_tools(user_message: str) -> str:
    """
    Call the Hugging Face-backed LLM via OpenAI client with function-calling tools.

    Steps:
    1. Build messages with system + history + new user message.
    2. Ask the model with tools/tool_choice="auto".
    3. If model calls tools, execute them in Python.
    4. Send tool results back to the model for a final, natural-language answer.
    5. Return the final reply as a string.
    """
    if not HF_API_KEY:
        log_event("HF_API_KEY (HF_TOKEN) is missing.")
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Please configure your Hugging Face API token in the .env file."
        )

    # Start from system message + prior LLM messages (tool calls, etc.)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(st.session_state.llm_messages)
    messages.append({"role": "user", "content": user_message})

    log_event(f"Sending user question to LLM: {user_message}")

    try:
        response = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
    except Exception as e:
        log_event(f"LLM error (first call): {e}")
        raise RuntimeError(f"Agent error (LLM first call): {e}")

    assistant_message = response.choices[0].message
    st.session_state.llm_messages.append({"role": "user", "content": user_message})
    st.session_state.llm_messages.append(
        {
            "role": assistant_message.role,
            "content": assistant_message.content,
            "tool_calls": getattr(assistant_message, "tool_calls", None),
        }
    )

    # If no tool calls -> direct answer
    if not assistant_message.tool_calls:
        final_answer = assistant_message.content or "I could not generate a response."
        log_event("LLM answered without calling tools.")
        return final_answer

    # Otherwise process each tool call
    tool_results = []

    for tool_call in assistant_message.tool_calls:
        tool_name = tool_call.function.name
        raw_args = tool_call.function.arguments or "{}"

        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            log_event(f"Failed to parse tool arguments: {raw_args}")
            args = {}

        log_event(f"Tool call received: {tool_name}({args})")

        tool_output = None
        error_message = None

        try:
            if tool_name == "run_sql_query":
                query = args.get("query", "")
                tool_output = safe_run_sql_query(query)
            elif tool_name == "get_table_overview":
                tool_output = get_table_overview()
            elif tool_name == "create_support_ticket":
                description = args.get("description", "")
                user_email = args.get("user_email")
                severity = args.get("severity", "medium")
                tool_output = create_support_ticket(description, user_email, severity)
            else:
                error_message = f"Unknown tool: {tool_name}"

        except Exception as e:
            error_message = f"Tool {tool_name} failed: {e}"

        if error_message:
            log_event(error_message)
            tool_output = {"error": error_message}

        # Add tool result into the conversation for the follow-up LLM call
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }
        )
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": json.dumps(tool_output),
            }
        )
        tool_results.append((tool_name, tool_output))

    # Second call: ask model to use the tool results to build final answer
    try:
        log_event("Sending tool results back to LLM for final answer.")
        followup = client.chat.completions.create(
            model=DEFAULT_LLM_MODEL,
            messages=messages,
        )
    except Exception as e:
        log_event(f"LLM error (second call): {e}")
        raise RuntimeError(f"Agent error (LLM second call): {e}")

    final_msg = followup.choices[0].message
    final_answer = final_msg.content or "I could not generate a response."

    st.session_state.llm_messages.append(
        {"role": "assistant", "content": final_answer}
    )

    return final_answer


# =========================
#  UI helpers: metrics & charts
# =========================

def load_sales_dataframe() -> pd.DataFrame:
    """Load the sales table into a pandas DataFrame for charts and metrics."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM sales", conn)
    conn.close()
    return df


def render_kpi_cards(df: pd.DataFrame):
    """Render some key KPI metrics at the top of the app."""
    total_revenue = df["total_amount"].sum()
    total_orders = df.shape[0]
    avg_order_value = df["total_amount"].mean()

    df["order_date"] = pd.to_datetime(df["order_date"])
    min_date = df["order_date"].min()
    max_date = df["order_date"].max()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"${total_revenue:,.0f}")
    c2.metric("Number of Orders", f"{total_orders:,}")
    c3.metric("Avg. Order Value", f"${avg_order_value:,.2f}")
    c4.metric("Date Range", f"{min_date.date()} → {max_date.date()}")


def render_charts(df: pd.DataFrame):
    """Render a simple revenue-by-region chart and revenue-by-product chart."""
    st.subheader("Quick Data Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Revenue by Region**")
        rev_region = df.groupby("region")["total_amount"].sum().reset_index()
        st.bar_chart(rev_region, x="region", y="total_amount")

    with col2:
        st.markdown("**Revenue by Product**")
        rev_product = df.groupby("product")["total_amount"].sum().reset_index()
        st.bar_chart(rev_product, x="product", y="total_amount")


# =========================
#  Main Streamlit app
# =========================

def main():
    st.set_page_config(
        page_title="Data Insights App – Capstone Project 1",
        page_icon="📊",
        layout="wide",
    )

    init_session_state()
    init_database()

    st.title("📊 Data Insights App – Capstone Project 1")
    st.caption(
        "Agent built with Python + Streamlit + Hugging Face (function calling with tools)."
    )

    # Load data for metrics & charts
    df = load_sales_dataframe()

    # Layout: main area + sidebar
    main_col, side_col = st.columns([3, 1])

    with main_col:
        render_kpi_cards(df)
        render_charts(df)

        st.markdown("---")
        st.subheader("💬 Ask the Data Assistant")

        # Chat history display
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask a question about sales data, KPIs, or trends…")

        if user_input:
            # Show user message immediately
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input}
            )
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_input)

            try:
                answer = call_agent_with_tools(user_input)
            except Exception as e:
                answer = f"Agent error: {e}"
                log_event(str(e))

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(answer)

    with side_col:
        st.subheader("ℹ️ App Info")
        st.markdown(
            """
**Data Insights App – Capstone Project 1**

- Backend: SQLite (≥500 sales records)
- UI: Streamlit
- LLM: Hugging Face Inference via OpenAI-compatible client
- Features:
  - Safe SQL analysis (read-only)
  - Function calling with tools
  - Support ticket creation
            """
        )

        st.markdown("---")
        st.subheader("📌 Sample Questions")
        st.markdown(
            """
- "Show total revenue by region."
- "Which product generated the highest revenue last month?"
- "How many orders did we have in the North region?"
- "What is the average order value per customer segment?"
- "I have a problem, please create a support ticket for me."
            """
        )

        st.markdown("---")
        st.subheader("🆘 Create Support Ticket (Manual)")
        ticket_text = st.text_area(
            "Describe your issue. The agent can also create tickets automatically.",
            height=100,
        )
        email_text = st.text_input("Your email (optional)")
        severity = st.selectbox("Severity", ["low", "medium", "high"], index=1)

        if st.button("Create Support Ticket"):
            if not ticket_text.strip():
                st.warning("Please describe your issue before creating a ticket.")
            else:
                try:
                    result = create_support_ticket(
                        description=ticket_text.strip(),
                        user_email=email_text.strip() or None,
                        severity=severity,
                    )
                    st.success(
                        f"Ticket #{result['ticket_id']} created with status '{result['status']}'."
                    )
                    log_event(
                        f"Manual support ticket created: #{result['ticket_id']} ({severity})"
                    )
                except Exception as e:
                    st.error(f"Failed to create ticket: {e}")
                    log_event(f"Support ticket creation error: {e}")

        st.markdown("---")
        st.subheader("📜 Logs (Last 50)")
        # Show only last 50 log lines for readability
        last_logs = st.session_state.logs[-50:]
        st.code("\n".join(last_logs) if last_logs else "No logs yet.")


if __name__ == "__main__":
    main()
