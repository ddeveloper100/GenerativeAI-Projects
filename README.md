# AI-Powered BI Dashboard

This repository contains projects developed to integrate AI capabilities into a Business Intelligence (BI) dashboard. The dashboard uses SQL queries to retrieve data from a database and visualize it through interactive charts and reports. The goal is to simplify data analysis by enabling natural-language querying, automated insights, and predictive analytics.

## Features

### 1. SQL Query Generator
Generates SQL queries from natural-language prompts. It uses contextual information about the database schema, tables, and columns to generate queries tailored to the underlying data, reducing the need to write SQL manually.

### 2. Automated Insights Generator
Analyzes the data represented in a dashboard to generate meaningful insights, highlight trends, and identify significant changes. This helps users quickly understand key findings without having to manually interpret every chart.

### 3. Predictive Analytics Model
Uses historical data from the dashboard to generate predictions about potential future trends. It also includes an interactive chat interface where users can ask questions about specific aspects of the dashboard and receive data-driven predictive insights.

## Tech Stack

- **Python** — Backend development and AI-powered data processing.
- **FastAPI** — Building API endpoints to connect the dashboard with the AI functionalities.
- **FAISS** — Efficient similarity search and retrieval of relevant database schema information.
- **Ollama** — Running and interacting with large language models (LLMs) for SQL generation and AI-powered insights.

## Objective

The objective of this project is to combine business intelligence with AI-driven analysis to make data exploration more accessible, automate insight generation, and support data-informed decision-making.
