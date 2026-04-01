import urllib.request
import xml.etree.ElementTree as ET
from textblob import TextBlob

def get_news_sentiment(ticker_symbol):
    """
    Fetches the latest news headlines from Yahoo Finance's RSS feed and calculates 
    an overall sentiment score using Natural Language Processing.
    """
    # Use Yahoo's official RSS feed, which is much more stable than yfinance's .news module
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker_symbol}&region=US&lang=en-US"
    
    try:
        # We must spoof a "User-Agent" so Yahoo thinks we are a standard web browser, not a bot
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            
        # Parse the XML data structure
        root = ET.fromstring(xml_data)
        items = root.findall('.//item')
    except Exception:
        return None, 0, "Neutral"

    if not items:
        return None, 0, "Neutral"

    articles = []
    total_polarity = 0

    # Loop through the latest 5 articles
    for item in items[:5]:
        # Extract data from the XML tags
        title = item.find('title').text if item.find('title') is not None else 'No Title'
        link = item.find('link').text if item.find('link') is not None else '#'
        pub_date = item.find('pubDate').text if item.find('pubDate') is not None else 'Recent'
        
        # Clean up the date string slightly (removes the timezone fluff at the end)
        if len(pub_date) > 22:
            pub_date = pub_date[:22]

        # NLP Magic: Calculate sentiment polarity of the headline
        analysis = TextBlob(title)
        polarity = analysis.sentiment.polarity
        total_polarity += polarity

        articles.append({
            'title': title,
            'publisher': 'Yahoo Finance News', # RSS aggregates all publishers here
            'link': link,
            'date': pub_date,
            'sentiment': polarity
        })

    # Calculate average sentiment across all recent articles
    avg_polarity = total_polarity / len(articles) if articles else 0
    
    # Translate the math into a human label
    if avg_polarity > 0.15:
        overall_mood = "Bullish / Positive"
    elif avg_polarity < -0.15:
        overall_mood = "Bearish / Negative"
    else:
        overall_mood = "Neutral / Mixed"

    return articles, avg_polarity, overall_mood