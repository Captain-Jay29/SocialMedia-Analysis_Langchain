import re
import spacy
from typing import Dict, Optional

# Load NLP model (Spacy's lightweight English model)
nlp = spacy.load("en_core_web_sm")

def extract_query_parameters(query: str) -> Dict[str, Optional[str]]:
    """
    Process the user query and extract relevant parameters like topic, location, 
    time range, sentiment, and platform preference.
    """
    doc = nlp(query)
    
    # Extract topic (Assuming first noun phrase is the main topic)
    topic = None
    for chunk in doc.noun_chunks:
        topic = chunk.text
        break  # Only take the first noun phrase as the topic
    
    # Extract location (Simple regex-based approach for now)
    location_match = re.search(r' in ([A-Za-z ]+)', query)
    location = location_match.group(1) if location_match else None
    
    # Extract time range
    time_keywords = ["past 24 hours", "last week", "last month", "last year"]
    time_range = None
    for keyword in time_keywords:
        if keyword in query:
            time_range = keyword
            break
    
    # Extract sentiment filter
    sentiment_keywords = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
    sentiment = None
    for key, value in sentiment_keywords.items():
        if key in query:
            sentiment = value
            break
    
    # Extract platform preference
    platform_keywords = ["Twitter", "Reddit", "Facebook"]
    platform = None
    for keyword in platform_keywords:
        if keyword.lower() in query.lower():
            platform = keyword
            break
    
    return {
        "topic": topic,
        "location": location,
        "time_range": time_range,
        "sentiment": sentiment,
        "platform": platform
    }

# Example usage
if __name__ == "__main__":
    sample_query = "Get positive Twitter posts about AI advancements in San Francisco last week"
    extracted_params = extract_query_parameters(sample_query)
    print(extracted_params)
