import pandas as pd
import datetime
import requests
import json
from googletrans import Translator
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

from deep_translator import GoogleTranslator
import logging

# TODO, SI funciona bien, refactorear para poder reutilizar algunas partes en otras funciones de sentyment

# Al ejecutar esta funcion sera necesario obtener un nuevo curl de la web y actualizar los headers.
# Lo que hago es copiar el dataset.csv a mano, renombrarlo y ejecutar la funcion sobre ese archivo.
def obtener_sentimiento_general(ruta_dataset):

    if not os.path.isfile(ruta_dataset):
        raise FileNotFoundError(f"El archivo {ruta_dataset} no existe.")

    dataset = pd.read_csv(ruta_dataset)

    if 'Sentimiento' not in dataset.columns:
        dataset['Sentimiento'] = ''
        
    if 'Tweets_Utilizados' not in dataset.columns:
        dataset['Tweets_Utilizados'] = 0

    headers = {
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
    
    analyzer = SentimentIntensityAnalyzer()

    for index, row in dataset.iterrows():

        if row['Sentimiento'] == 'pos' or row['Sentimiento'] == 'neg' or row['Sentimiento'] == 'neu':
            continue

        fecha_desde = pd.to_datetime(row['Open_time'])

        fecha_hasta = fecha_desde + datetime.timedelta(days=1)
        
        print(f"Procesando fecha desde: {fecha_desde}, fecha hasta: {fecha_hasta}")
        
        palabras_clave = "bitcoin cryptocurrency crypto CryptoNews"
        params = {
            'variables': json.dumps({
                'rawQuery': f'({palabras_clave}) until:{fecha_hasta.strftime("%Y-%m-%d")} since:{fecha_desde.strftime("%Y-%m-%d")}',
                'max_results': 100,
                'count': 100,
                'querySource': 'typed_query',
                'product': 'Latest',
                'min_faves': 100
            }),
            'features': json.dumps({
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
            }),
        }
        
        tweets_utilizados = 0
        sentiments = {'pos': 0, 'neg': 0, 'neu': 0}
        
        for page in range(5):  # Obtener un máximo de tres páginas
            response = requests.get('https://twitter.com/i/api/graphql/ummoVKaeoT01eUyXutiSVQ/SearchTimeline', headers=headers, params=params)
            
            # Para los casos de rate limit exceeded
            if response.status_code == 429:
                reset_timestamp = int(response.headers['X-Rate-Limit-Reset'])
                sleep_time = max(0, reset_timestamp - datetime.datetime.now().timestamp())
                print(f"Rate limit exceeded. Waiting for {sleep_time} seconds until reset.")
                time.sleep(sleep_time)
                response = requests.get('https://twitter.com/i/api/graphql/ummoVKaeoT01eUyXutiSVQ/SearchTimeline', headers=headers, params=params)
            elif response.status_code != 200:  # Si el error no es 429, loggearlo
                logging.error(f'HTTP error occurred: Status code {response.status_code}')
                continue
            
            response_json = response.json()
            tweets = response_json['data']['search_by_raw_query']['search_timeline']['timeline']['instructions'][0]
            
            if 'entries' in tweets:
                tweets = tweets['entries']
            else:
                tweets = []

            print(f"Tweets para la pagina {page}, total: {len(tweets)}")

            for tweet in tweets:
                if 'itemContent' not in tweet['content']:
                    # Siempre hay 2 registros de paginado en el response que deben ser obviados
                    continue
                
                tweet_text = tweet['content']['itemContent']['tweet_results']['result']['legacy']['full_text']
                translated_tweet = GoogleTranslator(source='auto', target='en').translate(tweet_text)
                # print(f"Tweet: {translated_tweet}")
                sentiment_scores = analyzer.polarity_scores(translated_tweet)
                print(sentiment_scores.items())
                
                max_sentiment = 'neu'
                if sentiment_scores['compound'] >= 0.05:
                    max_sentiment = 'pos'
                elif sentiment_scores['compound'] <= -0.05:
                    max_sentiment = 'neg'
                
                sentiments[max_sentiment] += 1
                tweets_utilizados += 1
            
            #TODO: Separar en otra funcion "get_next_page_token"
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
                                    params['variables'] = json.dumps({
                                        'rawQuery': f'({palabras_clave}) until:{fecha_hasta.strftime("%Y-%m-%d")} since:{fecha_desde.strftime("%Y-%m-%d")}',
                                        'max_results': 100,
                                        'count': 100,
                                        'querySource': 'typed_query',
                                        'product': 'Latest',
                                        'min_faves': 100,
                                        'cursor': content['value']
                                    })
                                else:
                                    break
                        elif 'entries' in pagination_entry:
                            entries = pagination_entry['entries']
                            if entries:
                                first_entry = entries[-1]
                                if 'content' in first_entry and 'value' in first_entry['content']:
                                    params['variables'] = json.dumps({
                                        'rawQuery': f'({palabras_clave}) until:{fecha_hasta.strftime("%Y-%m-%d")} since:{fecha_desde.strftime("%Y-%m-%d")}',
                                        'max_results': 100,
                                        'count': 100,
                                        'querySource': 'typed_query',
                                        'product': 'Latest',
                                        'min_faves': 100,
                                        'cursor': first_entry['content']['value']
                                    })
                                else:
                                    break

        print(sentiments)
        # Determine overall sentiment for the dataset
        overall_sentiment = max(sentiments, key=sentiments.get)
        dataset.at[index, 'Sentimiento'] = overall_sentiment
        dataset.at[index, 'Tweets_Utilizados'] = tweets_utilizados
        dataset.to_csv(ruta_dataset, index=False, float_format='%.8f')
    
    return dataset

# Calcular sentimiento general del mercado
# Supongamos que 'histórico_precio' es tu DataFrame de pandas con la columna 'Open_time'
sentimiento_general = obtener_sentimiento_general("/Users/mmarchetta/Desktop/Tesis-2024/dataset_sentimiento_general.csv")
