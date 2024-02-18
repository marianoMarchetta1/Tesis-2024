import os
import time
import talib;
from datetime import datetime, timedelta
import requests
import pandas as pd;
import numpy as np;
import pytz

#Obtengo toda la data cruda. Sera responsabilidad de otra funcion realizar las normalizaciones y limpiezas necesarias
def generar_dataset(interval, start_time, end_time, par="BTCUSDT", moneda="BTC", monedas_lideres=['BTCUSDT', 'ETHUSDT', 'BNBUSDT']):
    histórico_precio = obtener_historico_precio(interval, start_time, end_time, par)
    guardar_dataset_en_csv(histórico_precio, name="histórico_precio.csv")

    histórico_precio_influyentes = obtener_histórico_precio_influyentes(interval, start_time, end_time, monedas_lideres)
    guardar_dataset_en_csv(histórico_precio_influyentes, name="histórico_precio_influyentes.csv")

    indicadores_tecnicos = calcular_indicadores_tecnicos(histórico_precio)
    guardar_dataset_en_csv(indicadores_tecnicos, name="indicadores_tecnicos.csv")

    whale_alerts_binance = obtener_whale_alerts_binance(start_time, end_time, moneda, 1000, histórico_precio)
    guardar_dataset_en_csv(whale_alerts_binance, name="whale_alerts_binance.csv")

    
    ### Codigo a testear en conjunto, lo testie aislado y funciona bien, sino algunas fehcas aparecian con hora y duplicadas
    histórico_precio['Open_time'] = pd.to_datetime(histórico_precio['Open_time']).dt.date
    histórico_precio_influyentes['Open_time'] = pd.to_datetime(histórico_precio_influyentes['Open_time']).dt.date
    indicadores_tecnicos['Open_time'] = pd.to_datetime(indicadores_tecnicos['Open_time']).dt.date
    whale_alerts_binance['Open_time'] = pd.to_datetime(whale_alerts_binance['Open_time']).dt.date
    #####
    
    dataset = pd.merge(histórico_precio, histórico_precio_influyentes, on='Open_time', how='outer')
    dataset = pd.merge(dataset, indicadores_tecnicos, on='Open_time', how='outer')
    dataset = pd.merge(dataset, whale_alerts_binance, on='Open_time', how='outer')
    
    dataset = dataset[39:] # Elimino los primeros 39 días para evitar valores NaN en los indicadores técnicos

    guardar_dataset_en_csv(dataset)
    
    #Esta funcion directametne actualiza el archivo dataset.csv
    obtener_sentimiento_general("dataset.csv")
    
    sentimiento_moneda = obtener_sentimiento_moneda(moneda)
    sentimiento_individuos = obtener_sentimiento_individuos()

    return dataset

def obtener_sentimiento_moneda(moneda):
    # Lógica para obtener el sentimiento específico de la moneda
    # Retorna un diccionario {fecha: sentimiento}
    return

def obtener_sentimiento_individuos():
    # Lógica para obtener el sentimiento de individuos influyentes
    # Retorna un diccionario {fecha: sentimiento}
    return

# Lógica para obtener el precio de cierre, apertura, maximo y minimo de otras líderes como BTC, ETH y BNB
def obtener_histórico_precio_influyentes(interval, start_time, end_time, monedas_lideres):
    dfs = []

    for moneda in monedas_lideres:
        df = obtener_historico_precio(interval, start_time, end_time, moneda)
        df.rename(columns={
            'Open time': 'Open_time',
            'Open': f'Open_{moneda}',
            'High': f'High_{moneda}',
            'Low': f'Low_{moneda}',
            'Close': f'Close_{moneda}',
            'Volume': f'Volume_{moneda}',
            'Quote asset volume': f'Quote_asset_volume_{moneda}',
            'Number of trades': f'Number_of_trades_{moneda}'
        }, inplace=True)
        dfs.append(df)
    
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='Open_time', how='outer')

    return merged_df


###################################################################################################

# Lógica para calcular indicadores técnicos
def calcular_indicadores_tecnicos(historico_precio):
    
    close_prices = np.array(historico_precio['Close'], dtype=np.double)
    high_prices = np.array(historico_precio['High'], dtype=np.double)
    low_prices = np.array(historico_precio['Low'], dtype=np.double)
    
    # Media Móvil Simple (SMA)
    sma_20 = talib.SMA(close_prices, timeperiod=20)
    
    # Media Móvil Exponencial (EMA)
    ema_20 = talib.EMA(close_prices, timeperiod=20)
    
    # Bandas de Bollinger
    upper_band, middle_band, lower_band = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
    
    # Índice de Fuerza Relativa (RSI)
    rsi = talib.RSI(close_prices, timeperiod=14)
    
    # MACD (Convergencia/Divergencia de Medias Móviles)
    macd, signal, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    
    # ADX (Índice Direccional Promedio)
    adx = talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)
    
    # Estocástico
    slowk, slowd = talib.STOCH(high_prices, low_prices, close_prices, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    
    # Índice de Canal de Materias Primas 
    cci = talib.CCI(high_prices, low_prices, close_prices, timeperiod=14)

    
    # Crear un DataFrame para almacenar los indicadores técnicos
    indicadores = pd.DataFrame({
        'Open_time': historico_precio['Open_time'],
        'SMA_20': sma_20,
        'EMA_20': ema_20,
        'Upper_Band': upper_band,
        'Middle_Band': middle_band,
        'Lower_Band': lower_band,
        'RSI': rsi,
        'MACD': macd,
        'Signal': signal,
        'ADX': adx,
        'SlowK': slowk,
        'SlowD': slowd,
        'CCI': cci
    }, index=historico_precio.index)
    
    return indicadores

###################################################################################################

def obtener_aggregate_trades(symbol, startTime, endTime):
    url = "https://api.binance.com/api/v3/aggTrades"
    trades = []
    fromId = None

    while True:
        params = {"symbol": symbol, "limit": 1000}

        if startTime:
            params["startTime"] = startTime
            startTime = None

        if endTime and not fromId:
            params["endTime"] = endTime

        if fromId:
            params["fromId"] = fromId

        response = requests.get(url, params=params)
        if response.status_code == 200:
            trades_data = response.json()
            if not trades_data:
                break
            trades.extend(trades_data)
            fromId = trades_data[-1]['a']
            last_timestamp = trades_data[-1]['T']

            print(f"Último timestamp: {datetime.utcfromtimestamp(endTime / 1000)} Último timestamp obtenido: {datetime.utcfromtimestamp(last_timestamp / 1000)}")
            if endTime and last_timestamp > endTime:
                break
        else:
            print("Error al obtener trades:", response.status_code)
            break

    return trades

# Umbral equivale a cantidad de monedas
def obtener_whale_alerts_binance(start_time, end_time, moneda, umbral, precio_historico):
    fecha = start_time
    resultado = pd.DataFrame(columns=["Open_time", "Buy_1000x_high", "sell_1000x_high"])

    while fecha < end_time:
        trades = obtener_aggregate_trades(moneda, fecha, fecha + 86400000)
        if trades:
            fecha_formato = datetime.utcfromtimestamp(fecha / 1000).date()
            precio_historico_filtered = precio_historico[precio_historico['Open_time'].dt.date == fecha_formato]
            high_value = float(precio_historico_filtered['High'].max())
            
            print(f"Fecha sobre la que voy a calcular los whales: {fecha_formato}")
            print(f"Fecha del mayor precio para ese dia {precio_historico['Open_time'].dt.date}")
            print(f"Mayor precio para ese dia: {float(precio_historico_filtered['High'].max())}")
            
            buy_1000x_high = sum(1 for trade in trades if trade['m'] and float(trade['p']) * float(trade['q']) > high_value * umbral)
            sell_1000x_high = sum(1 for trade in trades if not trade['m'] and float(trade['p']) * float(trade['q']) > high_value * umbral)

            resultado.loc[len(resultado)] = [pd.to_datetime(fecha, unit='ms'), buy_1000x_high, sell_1000x_high]

        fecha += 86400000# Siguiente día en milisegundos

    return resultado


###################################################################################################

# Lógica para guardar el dataset en un archivo CSV usando pandas
def guardar_dataset_en_csv(dataset, name="dataset.csv"):
    pd.DataFrame(dataset).to_csv(name, index=False)
    return

###################################################################################################

def obtener_historico_precio(interval, start_time, end_time, binance_symbol):
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': binance_symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume', 'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])
        df['Open_time'] = pd.to_datetime(df['Open time'], unit='ms')
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
        df = df[['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', 'Number of trades']]
        return df
    else:
        print("Error al obtener los datos OHLCV de Binance.")
        return None

########################### Ejemplos de uso ###########################
# Establecer la fecha específica al 14 de febrero de 2024
fecha_especifica = datetime(2024, 2, 14)


binance_symbol = 'DOTUSDT'
interval = '1d'  # Obtener datos diarios
# Retroceder 2 años desde la fecha específica y convertir a milisegundos
# A mi me interesa tomar 730 dias (2 años), a eso le agrego 40 dias mas,
# porque algunos indicadores tecnicos necesitan hasta 33 dias previos para ser calculados,
# sino me sucede que los primeros dias de la serie tienen ciertos indicadores en NaN.
# A su vez, le agrego un dia mas a la resta por el formato UTC de la API de binance
margin_days = 40 # 40 días de margen para los indicadores tecnicos
wanted_previous_dates = 730 # 2 años
start_time = int((fecha_especifica - timedelta(days=(margin_days + wanted_previous_dates + 1))).timestamp() * 1000)
end_time = int((fecha_especifica + timedelta(days=(1))).timestamp() * 1000)

# obtener_historico_precio
# datos_candlestick = obtener_historico_precio(interval, start_time, end_time, binance_symbol)
# print(datos_candlestick)

# obtener_histórico_precio_influyentes
# datos_influentes = obtener_histórico_precio_influyentes(interval, start_time, end_time, ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
# print(datos_influentes)

# generar_dataset
# datos_completo = generar_dataset(interval, start_time, end_time, binance_symbol, binance_symbol, ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
# print(datos_completo)

# calcular_indicadores_tecnicos
# datos_candlestick = obtener_historico_precio(interval, start_time, end_time, binance_symbol)
# indicadores_tecnicos = calcular_indicadores_tecnicos(datos_candlestick)
# print(indicadores_tecnicos)


# Calcular aggregated_trades
# aggregated_trades = obtener_aggregate_trades(binance_symbol, start_time, end_time)
# print(aggregated_trades)

# Calcular obtener_whale_alerts_binance
# historico_precio = obtener_historico_precio(interval, start_time, end_time, binance_symbol)
# print(historico_precio)
# print(datetime.utcfromtimestamp(start_time / 1000))
# print(datetime.utcfromtimestamp(end_time / 1000))
# whale_alerts = obtener_whale_alerts_binance(start_time, end_time, binance_symbol, 1000, historico_precio)
# print(whale_alerts)


######## Codigo para testear los merge de los datasets aisladamente ################################
# histórico_precio = pd.read_csv("/Users/mmarchetta/Desktop/Tesis-2024/histórico_precio.csv")
# histórico_precio_influyentes = pd.read_csv("/Users/mmarchetta/Desktop/Tesis-2024/histórico_precio_influyentes.csv")
# indicadores_tecnicos = pd.read_csv("/Users/mmarchetta/Desktop/Tesis-2024/indicadores_tecnicos.csv")
# whale_alerts_binance = pd.read_csv("/Users/mmarchetta/Desktop/Tesis-2024/whale_alerts_binance.csv")

# histórico_precio['Open_time'] = pd.to_datetime(histórico_precio['Open_time']).dt.date
# histórico_precio_influyentes['Open_time'] = pd.to_datetime(histórico_precio_influyentes['Open_time']).dt.date
# indicadores_tecnicos['Open_time'] = pd.to_datetime(indicadores_tecnicos['Open_time']).dt.date
# whale_alerts_binance['Open_time'] = pd.to_datetime(whale_alerts_binance['Open_time']).dt.date

# dataset = pd.merge(histórico_precio, histórico_precio_influyentes, on='Open_time', how='outer')
# dataset = pd.merge(dataset, indicadores_tecnicos, on='Open_time', how='outer')
# dataset = pd.merge(dataset, whale_alerts_binance, on='Open_time', how='outer')
    
# dataset = dataset[39:] # Elimino los primeros 39 días para evitar valores NaN en los indicadores técnicos

# guardar_dataset_en_csv(dataset, "dataset.csv")
########################################################

### Imatacion del endpoint searchTimeline


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
                tweet_text = ''

                ##### TODO: separar en otra funcion
                if 'itemContent' not in tweet['content']:
                    continue # Siempre hay 2 registros de paginado en el response que deben ser obviados
                
                if 'tweet' in tweet and 'legacy' in tweet['tweet'] and 'full_text' in tweet['tweet']['legacy']:
                    tweet_text = tweet['tweet']['legacy']['full_text']
                elif 'content' in tweet and 'itemContent' in tweet['content'] and 'tweet_results' in tweet['content']['itemContent'] and 'result' in tweet['content']['itemContent']['tweet_results'] and 'legacy' in tweet['content']['itemContent']['tweet_results']['result'] and 'full_text' in tweet['content']['itemContent']['tweet_results']['result']['legacy']:
                    tweet_text = tweet['content']['itemContent']['tweet_results']['result']['legacy']['full_text']
                else:
                    # Omitir si no se encuentra el texto completo del tweet en ninguna estructura conocida
                    continue
                #####
                
                ######TODO: Separa en otra funcion
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
                #####
            
            ######TODO: Separar en otra funcion "get_next_page_token"
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
            #####

        # Determine overall sentiment for the dataset
        overall_sentiment = max(sentiments, key=sentiments.get)
        dataset.at[index, 'Sentimiento'] = overall_sentiment
        dataset.at[index, 'Tweets_Utilizados'] = tweets_utilizados
        dataset.to_csv(ruta_dataset, index=False, float_format='%.8f')
    
    return dataset

# Calcular sentimiento general del mercado
# Supongamos que 'histórico_precio' es tu DataFrame de pandas con la columna 'Open_time'
sentimiento_general = obtener_sentimiento_general("/Users/mmarchetta/Desktop/Tesis-2024/dataset_sentimiento_general.csv")
