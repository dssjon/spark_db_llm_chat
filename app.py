import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import create_spark_dataframe_agent
from pyspark.sql import SparkSession
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from faker import Faker
import random

# Initialize Spark
spark = SparkSession.builder.appName("Guests").getOrCreate()

# Initialize Faker
fake = Faker()

# NOTE: This data model is just made up for this demo and doesn't reflect any real data
food_preferences = ["vegan", "gluten-free", "dairy-free"," "]

seating_preferences = [
    " ",
    "prefers outside",
    "needs wheelchair access",
    "likes bar seating",
]
tag = [
    " ",
    "regular customer",
    "VIP",
    "food critic",
]

# Generate and insert a small amount of fake data for this demo
data = []
for _ in range(5):
    row = {
        "id": str(random.randint(1000, 9999)),
        "guest_name": fake.name(),
        "phone": fake.phone_number(),
        "email": fake.email(),
        "birthday": fake.date_of_birth(minimum_age=18, maximum_age=90),
        "notes": fake.text(max_nb_chars=200),
        "first_visit": fake.date_this_year(),
        "total_spend": round(random.uniform(200, 1000), 2),
        "total_visits": random.randint(1, 50),
        "last_visit": fake.date_this_year(),
        "food_preferences": random.choice(food_preferences),
        "seating_preferences": random.choice(seating_preferences),
        "tag": random.choice(tag),
        "visit_history": [random.randint(0, 7) for _ in range(24)]
    }
    data.append(row)

# Create DataFrame
df = spark.createDataFrame(data)

# NOTE: This requires a valid API key, initialized as an environment variable

# NOTE: gpt-3.5 hallucinates and produces poor results
#model = "gpt-3.5-turbo"
model = "gpt-4"
llm = ChatOpenAI(
    model=model,
    temperature=0,
    streaming=True,
)

agent = create_spark_dataframe_agent(llm=llm, df=df, verbose=True)

st.title(f"Guest Data Insights")
st.subheader(f"Retrieval Augmented Generation (Spark + Gpt)")
result_df = df.select("guest_name", "last_visit" ,"total_visits", "total_spend").limit(5)

label = "(Limit 5 rows shown):"
st.caption(label)
st.dataframe(df.toPandas(), column_config={
    "id": None,
       "birthday":None,
       "email": None,
       "first_visit": None,
       "food_preferences": None,
       "notes": None,
       "relationships": None,
       "seating_preferences": None,
       "phone": None,
        "visit_history": st.column_config.LineChartColumn(
            "visit_frequency", y_min=0, y_max=7
        ),
    },
)


if prompt := st.chat_input(
    "As a restaurant owner, what are useful insights you can tell me about my guests?"
):
    instruction = " Only include insights about average spend, visit and and food and seating preferences"
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(
            st.container(),
            max_thought_containers=20,
        )
        response = agent.run(prompt + instruction, callbacks=[st_callback])
        st.write(response)
