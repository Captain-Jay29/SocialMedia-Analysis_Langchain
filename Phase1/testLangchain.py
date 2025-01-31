from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

# Define an LLM (OpenAI's GPT model)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

# Define a sample tool (can be extended later)
def fetch_social_media_data(query: str):
    # Simulate a response with sample tweets or posts
    sample_data = [
        "AI is revolutionizing healthcare! 🏥 #MachineLearning",
        "New research on AGI by OpenAI! 🤖 #ArtificialIntelligence",
        "Self-driving cars are getting smarter with deep learning. 🚗💡"
    ]
    return f"Here are some social media posts about {query}: " + ", ".join(sample_data)


tool = Tool(name="SocialMediaFetcher", func=fetch_social_media_data, description="Fetches social media posts.")

# Initialize the agent with tools
agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Test the agent
response = agent.run("Get social media posts related to AI advancements")
print(response)
