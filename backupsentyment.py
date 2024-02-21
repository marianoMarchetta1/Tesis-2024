import pandas as pd
from datetime import datetime, timedelta
import requests
import json
from Preprocess.helpers import TWT_FEATURES, TWT_HEADERS, get_next_page_token, get_tweet_text, process_response, translate_and_get_sentyment
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

    headers = TWT_HEADERS

    for index, row in dataset.iterrows():

        if row['Sentimiento'] == 'pos' or row['Sentimiento'] == 'neg' or row['Sentimiento'] == 'neu':
            continue

        fecha_desde = pd.to_datetime(row['Open_time'])

        fecha_hasta = fecha_desde + timedelta(days=1)
        
        print(f"Procesando fecha desde: {fecha_desde}, fecha hasta: {fecha_hasta}")
        
        palabras_clave = "bitcoin cryptocurrency crypto CryptoNews"
        params = {
            'variables': json.dumps({
                'rawQuery': f'({palabras_clave}) until:{fecha_hasta.strftime("%Y-%m-%d")} since:{fecha_desde.strftime("%Y-%m-%d")}',
                'max_results': 100,
                'count': 100,
                'querySource': 'typed_query',
                'product': 'Latest',
                'min_faves': 250
            }),
            'features': TWT_FEATURES,
        }
        
        tweets_utilizados = 0
        sentiments = {'pos': 0, 'neg': 0, 'neu': 0}
        
        for page in range(10):  # Obtener un máximo de tres páginas
            response = requests.get('https://twitter.com/i/api/graphql/ummoVKaeoT01eUyXutiSVQ/SearchTimeline', headers=headers, params=params)
            has_sleept, errored = process_response(response)
            
            if has_sleept:
                response = requests.get('https://twitter.com/i/api/graphql/ummoVKaeoT01eUyXutiSVQ/SearchTimeline', headers=headers, params=params)
            elif errored:
                continue
            
            response_json = response.json()
            tweets = response_json['data']['search_by_raw_query']['search_timeline']['timeline']['instructions'][0]
            
            if 'entries' in tweets:
                tweets = tweets['entries']
            else:
                tweets = []

            print(f"Tweets para la pagina {page}, total: {len(tweets)}")

            for tweet in tweets:

                tweet_text = get_tweet_text(tweet)
                
                if not len(tweet_text) > 0:
                    continue
                
                sentiment_scores = translate_and_get_sentyment(tweet_text)
                
                max_sentiment = 'neu'
                if sentiment_scores['compound'] >= 0.05:
                    max_sentiment = 'pos'
                elif sentiment_scores['compound'] <= -0.05:
                    max_sentiment = 'neg'
                
                sentiments[max_sentiment] += 1
                tweets_utilizados += 1
            
            cursor = get_next_page_token(response_json)
            
            if not len(cursor) == 0:
                params['variables'] = json.dumps({
                    'rawQuery': f'({palabras_clave}) until:{fecha_hasta.strftime("%Y-%m-%d")} since:{fecha_desde.strftime("%Y-%m-%d")}',
                    'max_results': 100,
                    'count': 100,
                    'querySource': 'typed_query',
                    'product': 'Latest',
                    'min_faves': 100,
                    'cursor': cursor
                })
            else:
                break

        # Determino el sentimiento general y guardo variables utilies en el dataset
        overall_sentiment = max(sentiments, key=sentiments.get)
        dataset.at[index, 'Sentimiento'] = overall_sentiment
        dataset.at[index, 'Cantidad_post'] = sentiments['pos']
        dataset.at[index, 'Cantidad_neg'] = sentiments['neg']
        dataset.at[index, 'Cantidad_neu'] = sentiments['neu']
        dataset.at[index, 'Tweets_Utilizados'] = tweets_utilizados
        dataset.to_csv(ruta_dataset, index=False, float_format='%.8f')
    
    return dataset

# Calcular sentimiento general del mercado
# Supongamos que 'histórico_precio' es tu DataFrame de pandas con la columna 'Open_time'
sentimiento_general = obtener_sentimiento_general("/Users/mmarchetta/Desktop/Tesis-2024/dataset_sentimiento_general.csv")
