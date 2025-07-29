import requests
from huggingface_hub import InferenceClient

HF_TOKEN = "hugging face token"


def fetch_news(ticker):  # ticker - the company's name
    """
    Fetches the 5 most recent news articles about the given keyword (ticker).
    Returns a list of dictionaries with 'title' and 'summary'.
    """
    api_key = 'newsapi_key'
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': ticker,
        'sortBy': 'publishedAt',
        'pageSize': 5,
        'apiKey': api_key
    }
    response = requests.get(url, params=params)
    articles = response.json().get('articles', [])
    return [{"title": a.get("title", ""), "summary": a.get("description", "")} for a in articles[:5]]



def analyze_sentiment(text):
    client = InferenceClient(
        provider="hf-inference",
        api_key=HF_TOKEN
    )
    try:
        # Call FinBERT for financial sentiment analysis
        result = client.text_classification(text, model="ProsusAI/finbert")

        # Find the sentiment with highest confidence
        if result:
            top_sentiment = max(result, key=lambda x: x['score'])

            # Convert to your original format (Positive/Negative/Neutral)
            sentiment_label = top_sentiment['label'].lower()

            if sentiment_label == 'positive':
                return "Positive"
            elif sentiment_label == 'negative':
                return "Negative"
            else:
                return "Neutral"
        else:
            return "Neutral"  # Default if no result

    except Exception as e:
        print(f"FinBERT analysis failed: {e}")
        # Fallback to neutral if API fails
        return "Neutral"


def detect_signal(price_change, sentiment):
    import requests
    # Replace with your own Gemini API key
    GEMINI_API_KEY = "key"
    # Using Gemini 2.5 Flash Lite model endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={GEMINI_API_KEY}"
    # Instruction prompt for the model
    prompt = f"""
You are an expert financial trading assistant helping investors make decisions. 

Given the stock's recent price change and the market sentiment, determine the best trading action.

Only choose one of the following actions: Buy, Sell, or Hold.

Data:
- Price change: {price_change}%
- Sentiment: {sentiment}

Respond with only one word ("buy", "sell", or "hold") and nothing else.
""".strip()
    # Set headers and construct the request body
    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    try:
        # Send POST request to Gemini API
        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            # Parse the model's one-word response
            reply = response.json()['candidates'][0]['content']['parts'][0]['text'].strip().lower()

            # Normalize and return the decision
            if "buy" in reply:
                return "Buy"
            elif "sell" in reply:
                return "Sell"
            elif "hold" in reply:
                return "Hold"
            else:
                return "Hold"  # Default fallback if unexpected output
        else:
            
            return "Hold"  # Fallback on API error
    except Exception as e:
        return "Hold"  # Fallback on request failure

     
# nirvaan
def generate_newsletter(news_list, signals):
    newsletter = "*** Daily Financial Newsletter ***\n\n"

    # Add news section
    newsletter += "Top News Headlines:\n"
    for i, news_item in enumerate(news_list, start=1):
        newsletter += f"{i}. {news_item['title']}\n"
        newsletter += f"   {news_item['summary']}\n\n"

    # Add trading signals section
    newsletter += "Trading Signals:\n"
    for signal in signals:
        newsletter += f"- {signal}\n"

    return newsletter


# shree
def run_pipeline(ticker: str, price_change: float) -> str:
    # Step 1: Fetch news articles for the given ticker
    news_articles = fetch_news(ticker)

    # Step 2: Analyze sentiment for each news article title
    sentiments = []
    for article in news_articles:
        sentiment = analyze_sentiment(article["title"])
        sentiments.append(sentiment)

    # Step 3: Generate trading signals based on price change and sentiment
    signals = []
    for sentiment in sentiments:
        signal = detect_signal(price_change, sentiment)
        signals.append(signal)

    # Step 4: Generate the newsletter using the news articles and signals
    newsletter = generate_newsletter(news_articles, signals)

    # Step 5: Print and return the newsletter
    print(newsletter)
    return newsletter


if __name__ == "__main__":
    run_pipeline("TCS", 2.5)