from langchain_community.chat_models import ChatOpenAI  # Updated import
from typing import Dict, Optional
import json

class MisinformationDetector:
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3):
        """Initialize the ChatOpenAI model."""
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)

    def generate_prompt(self, text: str) -> str:
        """Generate a structured prompt for misinformation detection."""
        return f"""
        Analyze the following social media post and determine if it contains misinformation. 
        Consider factual accuracy, logical consistency, and cross-check with known facts. 
        Provide a boolean flag (True/False), a confidence score (0-1), and a short reason.
        
        Post: {text}
        
        Response format: JSON with keys: is_misinformation, confidence, reason.
        """

    def parse_response(self, response) -> Dict[str, any]:
        """Extract and parse JSON from LLM response."""
        if hasattr(response, "content"):  
            response_text = response.content  # Extract actual text
        else:
            return {"error": "Unexpected response format"}

        # Remove JSON formatting markers
        response_text = response_text.strip("```json").strip("```")

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse response"}

    def detect_misinformation(self, text: str) -> Dict[str, any]:
        """
        Uses an LLM to determine if the given text contains misinformation.
        Returns a dictionary containing:
            - is_misinformation (bool)
            - confidence (float)
            - reason (str)
        """
        prompt = self.generate_prompt(text)
        response = self.llm.invoke(prompt)
        return self.parse_response(response)

# Example Usage
if __name__ == "__main__":
    detector = MisinformationDetector()
    test_text = "COVID-19 vaccines contain microchips for tracking."
    result = detector.detect_misinformation(test_text)
    print(result)
