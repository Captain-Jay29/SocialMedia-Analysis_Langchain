import os
import tweepy
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("TWITTER_API_KEY")
API_SECRET = os.getenv("TWITTER_API_SECRET")
BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN")
ACCESS_TOKEN = os.getenv("TWITTER_ACCESS_TOKEN")
ACCESS_SECRET = os.getenv("TWITTER_ACCESS_SECRET")

# Authenticate
auth = tweepy.OAuth1UserHandler(API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_SECRET)
api = tweepy.API(auth)

# Verify authentication
try:
    user = api.verify_credentials()
    print(f"Authenticated as {user.screen_name}")
except Exception as e:
    print(f"Authentication failed: {e}")