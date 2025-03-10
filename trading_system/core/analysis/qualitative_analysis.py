import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
from textblob import TextBlob

# Configure logging
logger = logging.getLogger("trading_system.qualitative_analysis")

class APIClient:
    """Interface para APIs externas."""
    
    POLYGON_API_KEY = None
    
    @staticmethod
    def get_polygon_news(ticker: str, limit: int = 10) -> List[Dict]:
        """Obtém notícias relacionadas a um ticker via Polygon API."""
        if not APIClient.POLYGON_API_KEY:
            return []
        
        try:
            import requests
            url = "https://api.polygon.io/v2/reference/news"
            params = {"apiKey": APIClient.POLYGON_API_KEY, "ticker": ticker, "limit": limit, "order": "desc"}
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("results", [])
        except Exception as e:
            logger.error(f"Erro ao obter notícias da Polygon: {e}")
            return []

    @staticmethod
    def get_polygon_related(ticker: str) -> List[str]:
        """Obtém tickers relacionados via Polygon API."""
        if not APIClient.POLYGON_API_KEY:
            return []
        
        try:
            import requests
            url = f"https://api.polygon.io/vX/reference/tickers/{ticker}/relationships"
            params = {"apiKey": APIClient.POLYGON_API_KEY}
            resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code == 404:
                url = f"https://api.polygon.io/v3/reference/tickers/{ticker}"
                resp = requests.get(url, params=params, timeout=10)
            
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            results = []
            
            if "results" in data:
                if "related_tickers" in data["results"]:
                    results = data["results"]["related_tickers"]
                elif "brands" in data["results"]:
                    for brand in data["results"]["brands"]:
                        results.extend(brand.get("tickers", []))
            
            return list(set(results))
        except Exception as e:
            logger.error(f"Erro ao obter relacionados da Polygon para {ticker}: {e}")
            return []

class QualitativeAnalysis:
    """Análise de notícias e sentimento."""
    
    @staticmethod
    def analyze_news_sentiment(ticker: str) -> Dict[str, Any]:
        """Analisa o sentimento de notícias relacionadas a um ticker."""
        news_items = []
        
        # Tenta obter notícias via yfinance
        try:
            import yfinance as yf
            yf_news = yf.Ticker(ticker).get_news()
            if yf_news:
                for item in yf_news:
                    news_items.append({
                        "title": item.get("title", ""),
                        "published": item.get("publisher"),
                        "time": item.get("providerPublishTime", None),
                        "url": item.get("link", "")
                    })
            else:
                logger.info(f"Nenhuma notícia encontrada via yfinance para {ticker}.")
        except Exception as e:
            logger.error(f"Erro ao obter notícias via yfinance: {e}")
        
        # Tenta obter notícias via Polygon
        poly_items = APIClient.get_polygon_news(ticker, limit=5)
        for item in poly_items:
            news_items.append({
                "title": item.get("title", ""),
                "published": item.get("published_utc", ""),
                "time": item.get("timestamp", None),
                "url": item.get("article_url", "")
            })
        
        # Remove notícias duplicadas
        titles_seen = set()
        unique_news = []
        for item in news_items:
            title = item.get("title", "")
            if title and title not in titles_seen:
                titles_seen.add(title)
                unique_news.append(item)
        
        news_items = unique_news
        
        # Calcula o sentimento das notícias
        total_weighted_sent = 0.0
        total_weight = 0.0
        sentiment_details = []
        now = datetime.now(timezone.utc)
        
        for item in news_items:
            title = item.get("title", "")
            if not title:
                continue
            
            try:
                if item.get("published"):
                    if isinstance(item["published"], str):
                        pub_date = datetime.strptime(item["published"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                    else:
                        pub_date = datetime.fromtimestamp(item["time"], tz=timezone.utc)
                else:
                    pub_date = now
            except Exception:
                pub_date = now
            
            days_ago = max(0, (now - pub_date).days)
            weight = 1 / (1 + 0.5 * days_ago)
            
            polarity = TextBlob(title).sentiment.polarity
            if abs(polarity) > 0.5:
                polarity *= 1.2
            
            total_weighted_sent += polarity * weight
            total_weight += weight
            
            sentiment_details.append({
                "title": title,
                "date": pub_date.strftime("%Y-%m-%d"),
                "sentiment": float(polarity),
                "weight": round(weight, 2)
            })
        
        # Calcula o sentimento médio
        avg_sentiment = total_weighted_sent / total_weight if total_weight > 0 else 0.0
        
        if avg_sentiment > 0.3:
            sentiment_label = "muito positivo"
        elif avg_sentiment > 0:
            sentiment_label = "positivo"
        elif avg_sentiment < -0.3:
            sentiment_label = "muito negativo"
        elif avg_sentiment < 0:
            sentiment_label = "negativo"
        else:
            sentiment_label = "neutro"
        
        return {
            "sentiment_score": float(avg_sentiment),
            "sentiment_label": sentiment_label,
            "news_count": len(news_items),
            "details": sentiment_details
        }

    @staticmethod
    def qualitative_score(ticker: str) -> Dict[str, Any]:
        """Calcula um score qualitativo com base nas notícias e dados relacionados."""
        result = {"qualitative_score": 0, "details": {}}
        
        try:
            news_result = QualitativeAnalysis.analyze_news_sentiment(ticker)
            sent_score = news_result["sentiment_score"] * 20
            
            news_volume_bonus = min(10, news_result["news_count"])
            
            related_bonus = 0
            related_list = APIClient.get_polygon_related(ticker)
            if related_list:
                related_bonus = min(5, len(related_list))
            
            total = sent_score + news_volume_bonus + related_bonus
            
            result["qualitative_score"] = float(np.clip(total, -100, 100))
            result["details"] = {
                "sentiment_component": float(sent_score),
                "news_count": news_result["news_count"],
                "related_count": len(related_list),
                "sentiment_label": news_result["sentiment_label"],
                "news_samples": news_result["details"][:3] if len(news_result["details"]) >= 3 else news_result["details"]
            }
        except Exception as e:
            logger.error(f"Erro ao calcular qualitative_score para {ticker}: {e}")
            result["details"]["error"] = str(e)
        
        return result