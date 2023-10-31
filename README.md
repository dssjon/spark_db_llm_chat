# Retrieval augmented generation with spark db & gpt

This project utilizes Apache Spark and OpenAI's GPT API to generate insights about restaurant guests based on sample customer data.

#### Overview

- Data generated using Faker to create customer records with attributes like name, contact info, visit history, preferences, etc.
- The agent queries the Spark DataFrame and uses the GPT model to generate a natural language response with insights.

#### Key Components

- **Spark DataFrame**: Stores data for analysis
- **Streamlit**: Creates UI app for interactive queries
- **LangChain**: Sets up LLM agent with query access
- **OpenAI GPT API**: Provides access to GPT models like GPT-3. & GPT-4

#### Running the App

- Install requirements: pip install -r requirements.txt
- Run app: streamlit run app.py
- Ask agent questions in the Streamlit chat window
