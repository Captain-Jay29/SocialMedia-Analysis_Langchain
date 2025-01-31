import requests
import json
import psycopg2
from datetime import datetime

# Database Configuration
db_config = {
    "dbname": "social_media_db",
    "user": "jay",
    "host": "localhost",
    "port": "5432"
}

# Function to store raw data into PostgreSQL
def store_raw_data(data, platform):
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    
    insert_query = """
    INSERT INTO raw_posts (platform, post_id, username, content, location, timestamp, metadata)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (post_id) DO NOTHING;
    """
    
    for post in data:
        cursor.execute(insert_query, (
            platform,
            post.get("id"),
            post.get("username"),
            post.get("content"),
            post.get("location", "Unknown"),
            datetime.strptime(post.get("timestamp"), "%Y-%m-%dT%H:%M:%SZ"),
            json.dumps(post.get("metadata", {}))
        ))
    
    conn.commit()
    cursor.close()
    conn.close()

# Function to fetch data from Twitter API
def fetch_twitter_data(query, location=None, since=None, until=None):
    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": "Bearer YOUR_TWITTER_BEARER_TOKEN"}
    params = {
        "query": query,
        "tweet.fields": "created_at,author_id,geo",
        "max_results": 100
    }
    
    if location:
        params["place"] = location
    if since:
        params["start_time"] = since
    if until:
        params["end_time"] = until
    
    response = requests.get(url, headers=headers, params=params, timeout=60)
    
    if response.status_code == 200:
        tweets = response.json().get("data", [])
        processed_data = []
        for tweet in tweets:
            processed_data.append({
                "id": tweet["id"],
                "username": tweet["author_id"],
                "content": tweet["text"],
                "location": tweet.get("geo", {}).get("place_id", "Unknown"),
                "timestamp": tweet["created_at"],
                "metadata": tweet
            })
        
        store_raw_data(processed_data, "Twitter")
        return processed_data
    else:
        print(f"Error fetching data: {response.status_code} - {response.text}")
        return []

if __name__ == "__main__":
    query = "climate change"
    location = "New York"
    fetched_data = fetch_twitter_data(query, location)
    print(f"Fetched {len(fetched_data)} tweets")
