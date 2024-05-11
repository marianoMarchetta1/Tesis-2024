import json
import logging
import time
from deep_translator import GoogleTranslator
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
import requests
from coinbase import jwt_generator
from private_secrets import COINBASE_APY_KEY, COINBASE_APY_SECRET
from datetime import datetime, timezone
import pandas as pd

TWT_HEADERS = {
    'authority': 'twitter.com',
    'accept': '*/*',
    'accept-language': 'en-US,en;q=0.9,es;q=0.8',
    'authorization': 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA',
    'cache-control': 'no-cache',
    'content-type': 'application/json',
    'cookie': 'guest_id=v1%3A170146273370040632; guest_id_marketing=v1%3A170146273370040632; guest_id_ads=v1%3A170146273370040632; _ga=GA1.2.497459894.1708027635; external_referer=padhuUp37zi8sc5iOYsZ2B1wXDLbFndl|0|8e8t2xd8A2w%3D; g_state={"i_p":1708090431645,"i_l":1}; _twitter_sess=BAh7CSIKZmxhc2hJQzonQWN0aW9uQ29udHJvbGxlcjo6Rmxhc2g6OkZsYXNo%250ASGFzaHsABjoKQHVzZWR7ADoPY3JlYXRlZF9hdGwrCMnXsbGNAToMY3NyZl9p%250AZCIlZTU4NjU4OGFlMGRiZDcwMjIzNWVmMmVmMmE1ODFlNzQ6B2lkIiUxYWIx%250AZDc4MDFkZTZmZGFhMzgyN2MyMDUyOGQ1MzJkNg%253D%253D--0ec30f12c47eb0db41c2622cc3c5ac6fd37bab73; kdt=bB8HH6rk3WaYRhxbHb2EgynJZ0vu49BZ7h9w7CTQ; auth_token=44bed893e48e7599ff4cc092b02cfdcfec3e8f7b; ct0=6445e50e6dc5ab128740ac4e24904255326ee127a00bd4683e8bdd8e2004e501839465ff88a45aadb021743f1ad668a78c7aadfc036b8e529b44d38a5c33a715b965279e136fd357b4af76947aab01fd; lang=es; twid=u%3D4527701956; _gid=GA1.2.1446814894.1708178504; personalization_id="v1_ssQSRtcVYNkXSgLMDWGhUg=="',
    'pragma': 'no-cache',
    'referer': 'https://twitter.com/search?q=(bitcoin%20eth%20btc)%20until%3A2023-01-02%20since%3A2023-01-01&src=typed_query&f=live',
    'sec-ch-ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"macOS"',
    'sec-fetch-dest': 'empty',
    'sec-fetch-mode': 'cors',
    'sec-fetch-site': 'same-origin',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
    'x-client-transaction-id': '8HIpCybPrULHe/6ICpk1oZTGQtEnX9PLAGAJgpzztDuVqV1o7Gu2nVP2tK9st219QTxfcvEB0qykLcXyCG0MeBG+4JMS8Q',
    'x-csrf-token': '6445e50e6dc5ab128740ac4e24904255326ee127a00bd4683e8bdd8e2004e501839465ff88a45aadb021743f1ad668a78c7aadfc036b8e529b44d38a5c33a715b965279e136fd357b4af76947aab01fd',
    'x-twitter-active-user': 'yes',
    'x-twitter-auth-type': 'OAuth2Session',
    'x-twitter-client-language': 'es',
}

TWT_FEATURES = json.dumps({
    'responsive_web_graphql_exclude_directive_enabled': True,
    'verified_phone_label_enabled': False,
    'creator_subscriptions_tweet_preview_api_enabled': True,
    'responsive_web_graphql_timeline_navigation_enabled': True,
    'responsive_web_graphql_skip_user_profile_image_extensions_enabled': False,
    'c9s_tweet_anatomy_moderator_badge_enabled': True,
    'tweetypie_unmention_optimization_enabled': True,
    'responsive_web_edit_tweet_api_enabled': True,
    'graphql_is_translatable_rweb_tweet_is_translatable_enabled': True,
    'view_counts_everywhere_api_enabled': True,
    'longform_notetweets_consumption_enabled': True,
    'responsive_web_twitter_article_tweet_consumption_enabled': True,
    'tweet_awards_web_tipping_enabled': False,
    'freedom_of_speech_not_reach_fetch_enabled': True,
    'standardized_nudges_misinfo': True,
    'tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled': True,
    'rweb_video_timestamps_enabled': True,
    'longform_notetweets_rich_text_read_enabled': True,
    'longform_notetweets_inline_media_enabled': True,
    'responsive_web_enhance_cards_enabled': False
})

def get_tweet_text(tweet):
    if 'itemContent' not in tweet['content']:
        return ''
    
    if 'tweet' in tweet and 'legacy' in tweet['tweet'] and 'full_text' in tweet['tweet']['legacy']:
        return tweet['tweet']['legacy']['full_text']
    elif 'content' in tweet and 'itemContent' in tweet['content'] and 'tweet_results' in tweet['content']['itemContent'] and 'result' in tweet['content']['itemContent']['tweet_results'] and 'legacy' in tweet['content']['itemContent']['tweet_results']['result'] and 'full_text' in tweet['content']['itemContent']['tweet_results']['result']['legacy']:
        return tweet['content']['itemContent']['tweet_results']['result']['legacy']['full_text']
    else:
        # Omitir si no se encuentra el texto completo del tweet en ninguna estructura conocida
        return ''

def translate_and_get_sentyment(tweet_text):
    analyzer = SentimentIntensityAnalyzer()

    translated_tweet = GoogleTranslator(source='auto', target='en').translate(tweet_text)
    
    #Algunos tweets solamente contienen emoji (como consecuence de un repost), estos no tiene traducción
    if translated_tweet is None:
        return {'compound': 0.0}
    
    # print(f"Tweet: {translated_tweet}")
    sentiment_scores = analyzer.polarity_scores(translated_tweet)
    # print(sentiment_scores.items())
    return sentiment_scores

def get_next_page_token(response_json):

    if 'data' in response_json and 'search_by_raw_query' in response_json['data'] and 'search_timeline' in response_json['data']['search_by_raw_query']:
        search_timeline = response_json['data']['search_by_raw_query']['search_timeline']
        if 'timeline' in search_timeline:
            timeline = search_timeline['timeline']
            if 'instructions' in timeline and len(timeline['instructions']) > 0:
                instructions = timeline['instructions']
                pagination_entry = instructions[-1]
                if 'entry' in pagination_entry:
                    entry = pagination_entry['entry']
                    if 'content' in entry:
                        content = entry['content']
                        if 'value' in content:
                            return content['value']
                        else:
                            return ''
                elif 'entries' in pagination_entry:
                    entries = pagination_entry['entries']
                    if entries:
                        first_entry = entries[-1]
                        if 'content' in first_entry and 'value' in first_entry['content']:
                            return first_entry['content']['value']
                        else:
                            return ''
                        
    return ''

def process_response(response):
    # Para los casos de rate limit exceeded
    if response.status_code == 429:
        reset_timestamp = int(response.headers['X-Rate-Limit-Reset'])
        sleep_time = max(0, reset_timestamp - datetime.now().timestamp())
        print(f"Rate limit exceeded. Waiting for {sleep_time} seconds until reset.")
        time.sleep(sleep_time)
        return True, False
    elif response.status_code != 200:  # Si el error no es 429, loggearlo
        logging.error(f'HTTP error occurred: Status code {response.status_code}')
        return False, True
    
    return False, False

def get_tweet_max_sentyment(sentiment_scores):
    max_sentiment = 'neu'
    if sentiment_scores['compound'] >= 0.05:
        max_sentiment = 'pos'
    elif sentiment_scores['compound'] <= -0.05:
        max_sentiment = 'neg'
        
    return max_sentiment

# En vez de usar el compound (Valor normalizado) de cada tweet para promediar luego,
# la cantidad de twts positivos, negativos o neutros. Sumarizo el sentimiento de cada tweet
# y luego calculo el promedio del compound, para determinar cual es el sentimiento general.
def get_overall_sentyment(total_compound_score, tweets_utilizados):
    overall_sentiment = 'neu'
    average_compound_score = total_compound_score / tweets_utilizados
    if average_compound_score >= 0.05:
        overall_sentiment = 'pos'
    elif average_compound_score <= -0.05:
        overall_sentiment = 'neg'
    
    return overall_sentiment

# Obtiene los terminos relacionados a la moneda para luego buscar en redes sociales
def get_coin_related_terms(par):
    url = f"https://api.binance.com/api/v3/exchangeInfo"
    params = { "symbol": par }
    response = requests.get(url, params)
    par_data = response.json()
    baseAsset = par_data["symbols"][0]["baseAsset"]
    
    url_coins_list = f"https://api.coingecko.com/api/v3/coins/list"
    response_coins_list = requests.get(url_coins_list)
    coins_list = response_coins_list.json()
    coin_information = None
    
    for objeto in coins_list:
        if objeto.get("symbol") == baseAsset.lower() and objeto.get("name").lower() == objeto.get("id").lower():
            coin_information = objeto
            break

    coin_raled_terms = None
    coin_id = coin_information["id"]

    url_coin_related_terms = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    response_coin_related_terms = requests.get(url_coin_related_terms)
    coin_raled_terms = response_coin_related_terms.json()
    
    related_terms = [coin_raled_terms["name"], coin_raled_terms["symbol"], coin_raled_terms["id"], coin_raled_terms["web_slug"], f"#{coin_raled_terms['symbol']}", f"#{coin_raled_terms['name']}", f"#{coin_raled_terms['id']}"]
    
    return related_terms

def build_jwt_token(request_path):
    api_key = COINBASE_APY_KEY
    api_secret = COINBASE_APY_SECRET

    request_method = "GET"
    jwt_uri = jwt_generator.format_jwt_uri(request_method, request_path)
    jwt_token = jwt_generator.build_rest_jwt(jwt_uri, api_key, api_secret)
    
    return jwt_token


def parse_datetime_string(datetime_str):
    formats = ["%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S.%fZ", "%Y-%m-%d %H:%M:%SZ"]
    for fmt in formats:
        try:
            return datetime.strptime(datetime_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError("No se pudo analizar la cadena de fecha y hora")

def join_datasets(ruta_dataset1, ruta_dataset2):
    dataset1 = pd.read_csv(ruta_dataset1)
    dataset2 = pd.read_csv(ruta_dataset2)
    
    merged_dataset = pd.concat([dataset1, dataset2], ignore_index=True)
    
    merged_dataset.to_csv(ruta_dataset2, index=False, float_format='%.8f')