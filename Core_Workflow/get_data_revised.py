"""
File: get_data_revised.py

This module provides an integrated flow for:
1. Fetching URLs (via a search query or provided URL list)
2. Scraping clean content from each URL using the Jina Reader API
3. Cleaning the scraped content and summarizing it using a Hugging Face BART model
4. Returning a structured output (list of dictionaries) and optionally writing it to file

Usage (importing into your agent orchestrator):
    from get_data_revised import get_data_and_summarize

    # Using a search query:
    results = get_data_and_summarize(query="latest news on AI", num_results=5)

    # Or providing a list of URLs:
    results = get_data_and_summarize(url_list=["https://example.com/article1", "https://example.com/article2"])

Each result is a dictionary:
    {
        "url": <url>,
        "raw_content": <raw scraped content>,
        "summary": <summarized text>
    }
"""

import os
import re
import html
import requests
from urllib.parse import quote
from googlesearch import search
from bs4 import BeautifulSoup
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from dotenv import load_dotenv
import nltk

# Load environment variables and NLTK data
load_dotenv()
nltk.download('punkt', quiet=True)

###############################
# Configuration & Constants
###############################

API_KEY = os.getenv("Jina_API_KEY")
EXCLUDED_DOMAINS = ['youtube.com', 'youtu.be', 'vimeo.com', 'dailymotion.com']
DEBUG = True
MAX_RETRIES = 2  # For GPU error recovery

###############################
# Helper Debug Function
###############################

def debug_print(*args):
    if DEBUG:
        print("[DEBUG]", *args)

###############################
# Data Retrieval Functions
###############################

def get_urls(query, num_results=7):
    """
    Use googlesearch to fetch a list of URLs for the given query,
    excluding domains from video streaming platforms.
    """
    raw_urls = list(search(query, num_results=num_results * 2))
    filtered_urls = []
    for url in raw_urls:
        if any(domain in url for domain in EXCLUDED_DOMAINS):
            continue
        filtered_urls.append(url)
        if len(filtered_urls) >= num_results:
            break
    return filtered_urls

def get_clean_content(url):
    """
    Fetch clean content for a given URL using the Jina Reader API.
    """
    encoded_url = quote(url, safe='')
    api_endpoint = f"https://r.jina.ai/{encoded_url}"
    
    # headers = {
    #     "Authorization": f"Bearer {API_KEY}",
    #     "X-Engine": "direct"
    # }
    
    # response = requests.get(api_endpoint, headers=headers)
    response = requests.get(api_endpoint, timeout=60)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

def fetch_data(mode="query", query=None, url_list=None, num_results=5):
    """
    Fetches data from URLs either by performing a Google search using the provided query,
    or by using a given list of URLs.
    
    Returns a list of dictionaries:
        [
            {"url": <url>, "raw_content": <content>},
            ...
        ]
    """
    results = []
    if mode == "query":
        if not query:
            raise ValueError("Query must be provided when mode is 'query'.")
        urls = get_urls(query, num_results=num_results)
    elif mode == "urls":
        if not url_list:
            raise ValueError("url_list must be provided when mode is 'urls'.")
        urls = url_list
    else:
        raise ValueError("Invalid mode specified. Use 'query' or 'urls'.")
    
    for url in urls:
        try:
            content = get_clean_content(url)
            results.append({"url": url, "raw_content": content})
        except Exception as e:
            debug_print(f"Error processing {url}: {str(e)}")
            results.append({"url": url, "raw_content": f"Error: {str(e)}"})
    return results

###############################
# Summarization Functions
###############################

def clean_content(text):
    """
    Remove HTML/Markdown tags and clean text for summarization.
    """
    text = html.unescape(text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # Remove markdown images
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)    # Remove markdown links
    text = re.sub(r"#{1,6}\s*", "", text)          # Remove headers
    text = re.sub(r"\n+", ". ", text)              # Replace newlines with periods
    text = re.sub(r"\s+", " ", text)               # Remove extra spaces
    return text.strip()

def initialize_model():
    """
    Initialize and return a Hugging Face summarization pipeline using facebook/bart-large-cnn.
    """
    debug_print("Initializing model...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
        device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
        debug_print(f"Moving model to {device_str}")
        model = model.to(device_str)
        device = 0 if device_str.startswith("cuda") else -1
        summarizer = pipeline(
            "summarization",
            model=model,
            tokenizer=tokenizer,
            device=device,
        )
        return summarizer
    except Exception as e:
        debug_print("Model initialization failed:", e)
        raise

def safe_summarize(summarizer, content):
    """
    Summarize content using the provided summarizer with error recovery.
    Uses dynamic length checks based on the input.
    """
    for attempt in range(MAX_RETRIES):
        try:
            cleaned = clean_content(content)
            # Tokenize to check input length
            tokens = summarizer.tokenizer.encode(cleaned, truncation=False, add_special_tokens=False)
            # If input is very long, truncate to a safe limit (e.g., 1000 tokens)
            if len(tokens) > 1024:
                debug_print(f"Truncating content from {len(tokens)} tokens")
                cleaned = summarizer.tokenizer.decode(tokens[:1000], skip_special_tokens=True)
            # Set dynamic parameters based on input length
            input_length = len(cleaned)
            if input_length < 50:
                return cleaned  # Too short to summarize
            max_len = min(150, input_length // 2)
            min_len = max(10, max_len // 3)
            summary = summarizer(
                cleaned,
                max_length=max(30, max_len),
                min_length=min_len,
                do_sample=False,
                truncation=True
            )[0]['summary_text']
            return summary
        except RuntimeError as e:
            if "CUDA" in str(e) and attempt < MAX_RETRIES - 1:
                debug_print(f"GPU error on attempt {attempt+1}, retrying...")
                torch.cuda.empty_cache()
                continue
            raise
    return "Summary error: Maximum retries exceeded"

def process_data_for_summarization(summarizer, data):
    """
    For each entry in the data list (each a dict with 'url' and 'raw_content'),
    generate a summary using safe_summarize.
    
    Returns a new list of dictionaries with added key 'summary'.
    """
    results = []
    for entry in data:
        url = entry.get("url", "Unknown URL")
        raw_content = entry.get("raw_content", "")
        if raw_content.startswith("Error:"):
            summary = raw_content
        else:
            try:
                summary = safe_summarize(summarizer, raw_content)
            except Exception as e:
                summary = f"Summarization failed: {str(e)}"
        results.append({
            "url": url,
            "raw_content": raw_content,
            "summary": summary
        })
    return results

###############################
# Main Integration Function
###############################

def get_data_and_summarize(query=None, url_list=None, num_results=5, output_file=None):
    """
    Main function to fetch data and summarize it.
    
    Parameters:
        query (str): A search query to fetch URLs (if provided).
        url_list (list): A list of URLs to process (if provided).
        num_results (int): Number of URLs to fetch when using a query.
        output_file (str): Optional file path to write the structured output.
    
    Returns:
        results (list): A list of dictionaries with keys: 'url', 'raw_content', 'summary'
    """
    # Determine mode based on parameters
    mode = "query" if query else "urls"
    data = fetch_data(mode=mode, query=query, url_list=url_list, num_results=num_results)
    summarizer = initialize_model()
    results = process_data_for_summarization(summarizer, data)
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Write structured output as a simple text dump (could be JSON if desired)
                for entry in results:
                    f.write(f"URL: {entry['url']}\n")
                    f.write("-" * 50 + "\n")
                    f.write("Summary:\n")
                    f.write(entry['summary'] + "\n")
                    f.write("=" * 80 + "\n\n")
            debug_print(f"Results written to {output_file}")
        except Exception as e:
            debug_print(f"Failed to write to {output_file}: {e}")
    
    return results

###############################
# Optional: Main Block for Testing
###############################

if __name__ == "__main__":
    # For testing purposes, you can call this script directly.
    # Here we use a hardcoded query; in production, the agent would pass parameters.
    test_query = input("Enter a search query for testing: ").strip()
    results = get_data_and_summarize(query=test_query, num_results=5, output_file="structured_output.txt")
    print(f"\nProcessed {len(results)} results. Check 'structured_output.txt' for output.\n")
