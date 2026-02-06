GenerativeAI_Capstone1 – Data Insights App
Project Description

Data Insights App is an AI-powered application that allows users to query a structured database using natural language.
The agent retrieves information via controlled database queries and sends only partial results to the LLM.
The application also provides basic business insights and allows escalation to human support via support tickets.

Architecture Diagram
User
 │
 ▼
Streamlit UI
 │
 ├─ Chat Interface
 ├─ Business Metrics (row count, charts)
 │
 ▼
AI Agent (Function Calling)
 │
 ├─ Database Query Tool (read-only)
 ├─ Support Ticket Tool
 │
 ▼
Database (500+ rows)

How It Works

User enters a question in the chat

Agent analyzes the request

Agent calls the database tool (SELECT-only)

Query results are returned to the agent

Only the result subset is passed to the LLM

Response is shown to the user

Agent may suggest creating a support ticket if needed

Screenshots

(Add real screenshots from usage)

Screenshot 1: Application home page

Screenshot 2: Chat interaction

Screenshot 3: Business insights panel

Screenshot 4: Support ticket creation

Installation Steps
git clone https://github.com/azizkuchkarov/GenerativeAI_Capstone1
cd GenerativeAI_Capstone1
pip install -r requirements.txt

Usage Steps
streamlit run app.py


Open the app in the browser

Review dataset overview

Ask questions in the chat

View answers and insights

<img width="1866" height="992" alt="image" src="https://github.com/user-attachments/assets/7d62874a-3c26-466a-9b5f-96bde5d7888a" />


Create a support ticket if required

<img width="1842" height="773" alt="image" src="https://github.com/user-attachments/assets/62aa543f-0c3a-4407-9f08-b60de84ddd40" />


Safety Description

Only SELECT queries are allowed

DELETE, UPDATE, DROP operations are blocked

Query results are limited

Full dataset is never sent to the LLM

Unsafe actions are refused by the agent

Function-Calling Documentation

The agent uses function calling with at least two tools:

1. Database Query Function

Executes safe, read-only queries

Returns aggregated or filtered results

2. Support Ticket Function

Creates a support ticket when requested

Sends user context and issue summary

Support Ticket Feature

The agent can create support tickets when:

User explicitly requests human help

The question cannot be answered confidently

<img width="1871" height="933" alt="image" src="https://github.com/user-attachments/assets/a61da06c-01cd-46a5-a18a-a73d5ccbf157" />


Author

Aziz Kuchkarov
https://github.com/azizkuchkarov
