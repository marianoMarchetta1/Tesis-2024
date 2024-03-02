import os
import time
from helpers import TWT_FEATURES, TWT_HEADERS, build_jwt_token, get_coin_related_terms, get_next_page_token, get_overall_sentyment, get_tweet_max_sentyment, get_tweet_text, parse_datetime_string, process_response, translate_and_get_sentyment
import talib;
from datetime import datetime, timedelta, timezone
import requests
import pandas as pd;
import numpy as np;
import json

#Obtengo toda la data cruda. Sera responsabilidad de otra funcion realizar las normalizaciones y limpiezas necesarias
def generar_dataset(interval, start_time, end_time, start_date, end_date, coinbase_symbol="BTC-USDT", par="BTCUSDT", monedas_lideres=['BTCUSDT', 'ETHUSDT', 'BNBUSDT']):
    histórico_precio = obtener_historico_precio(interval, start_time, end_time, par)
    guardar_dataset_en_csv(histórico_precio, name="histórico_precio.csv")

    histórico_precio_influyentes = obtener_histórico_precio_influyentes(interval, start_time, end_time, monedas_lideres)
    guardar_dataset_en_csv(histórico_precio_influyentes, name="histórico_precio_influyentes.csv")

    indicadores_tecnicos = calcular_indicadores_tecnicos(histórico_precio)
    guardar_dataset_en_csv(indicadores_tecnicos, name="indicadores_tecnicos.csv")

    whale_alerts_binance = obtener_whale_alerts_binance(start_time, end_time, par, 1000, histórico_precio)
    guardar_dataset_en_csv(whale_alerts_binance, name="whale_alerts_binance.csv")

    whale_alerts_coinbase = obtener_whale_alerts_coinbase(start_date, end_date, coinbase_symbol, 1000, histórico_precio)
    guardar_dataset_en_csv(whale_alerts_coinbase, name="whale_alerts_coinbase.csv")
    
    ### Codigo a testear en conjunto, lo testie aislado y funciona bien, sino algunas fehcas aparecian con hora y duplicadas
    histórico_precio['Open_time'] = pd.to_datetime(histórico_precio['Open_time']).dt.date
    histórico_precio_influyentes['Open_time'] = pd.to_datetime(histórico_precio_influyentes['Open_time']).dt.date
    indicadores_tecnicos['Open_time'] = pd.to_datetime(indicadores_tecnicos['Open_time']).dt.date
    whale_alerts_binance['Open_time'] = pd.to_datetime(whale_alerts_binance['Open_time']).dt.date
    whale_alerts_coinbase['Open_time'] = pd.to_datetime(whale_alerts_coinbase['Open_time']).dt.date
    #####
    
    dataset = pd.merge(histórico_precio, histórico_precio_influyentes, on='Open_time', how='outer')
    dataset = pd.merge(dataset, indicadores_tecnicos, on='Open_time', how='outer')
    dataset = pd.merge(dataset, whale_alerts_binance, on='Open_time', how='outer')
    dataset = pd.merge(dataset, whale_alerts_coinbase, on='Open_time', how='outer')
    
    dataset = dataset[39:] # Elimino los primeros 39 días para evitar valores NaN en los indicadores técnicos

    guardar_dataset_en_csv(dataset)
    
    #Esta funcion directametne actualiza el archivo dataset.csv
    obtener_sentimiento_general("dataset.csv")
    obtener_sentimiento_moneda(par, "dataset.csv")
    obtener_sentimiento_individuos("dataset.csv")

    return dataset


###################################################################################################
# Funcion que encapsula la logica repetida de las otras funciones de sentyment
def obtener_sentimiento(ruta_dataset, palabras_clave, column_prefix=None, specific_authors=None):
    if not os.path.isfile(ruta_dataset):
        raise FileNotFoundError(f"El archivo {ruta_dataset} no existe.")

    dataset = pd.read_csv(ruta_dataset)

    sentimiento_column = 'Sentimiento'
    tweets_utilizados_column = 'Tweets_Utilizados'

    if column_prefix:
        sentimiento_column += f"_{column_prefix}"
        tweets_utilizados_column += f"_{column_prefix}"

    if sentimiento_column not in dataset.columns:
        dataset[sentimiento_column] = ''

    if tweets_utilizados_column not in dataset.columns:
        dataset[tweets_utilizados_column] = 0

    headers = TWT_HEADERS

    for index, row in dataset.iterrows():

        if row[sentimiento_column] == 'pos' or row[sentimiento_column] == 'neg' or row[sentimiento_column] == 'neu':
            continue

        fecha_desde = pd.to_datetime(row['Open_time'])

        fecha_hasta = fecha_desde + timedelta(days=1)

        print(f"Procesando fecha desde: {fecha_desde}, fecha hasta: {fecha_hasta}")

        query_authors = ''
        if specific_authors:
            query_authors = f" (from:{' OR from:'.join(specific_authors)})"

        params = {
            'variables': json.dumps({
                'rawQuery': f'({palabras_clave}){query_authors} until:{fecha_hasta.strftime("%Y-%m-%d")} since:{fecha_desde.strftime("%Y-%m-%d")}',
                'max_results': 100,
                'count': 100,
                'querySource': 'typed_query',
                'product': 'Latest',
                'min_faves': 250
            }),
            'features': TWT_FEATURES,
        }

        tweets_utilizados = 0
        total_compound_score = 0

        sentiments = {'pos': 0, 'neg': 0, 'neu': 0}

        for page in range(10):  # Obtener un máximo de tres páginas
            response = requests.get('https://twitter.com/i/api/graphql/ummoVKaeoT01eUyXutiSVQ/SearchTimeline',
                                    headers=headers, params=params)
            has_sleept, errored = process_response(response)

            if has_sleept:
                response = requests.get('https://twitter.com/i/api/graphql/ummoVKaeoT01eUyXutiSVQ/SearchTimeline',
                                        headers=headers, params=params)
            elif errored:
                continue

            response_json = response.json()
            tweets = response_json['data']['search_by_raw_query']['search_timeline']['timeline']['instructions'][0]

            if 'entries' in tweets:
                tweets = tweets['entries']
            else:
                tweets = []

            print(f"Tweets para la pagina {page}, total: {len(tweets)}")

            if len(tweets) == 0:
                break

            for tweet in tweets:
                tweet_text = get_tweet_text(tweet)

                if not len(tweet_text) > 0:
                    continue

                sentiment_scores = translate_and_get_sentyment(tweet_text)

                # Solo con propositos informativos
                max_tweet_sentyment = get_tweet_max_sentyment(sentiment_scores)
                total_compound_score += sentiment_scores['compound']
                sentiments[max_tweet_sentyment] += 1
                tweets_utilizados += 1

            cursor = get_next_page_token(response_json)

            if not len(cursor) == 0:
                params['variables'] = json.dumps({
                    'rawQuery': f'({palabras_clave}){query_authors} until:{fecha_hasta.strftime("%Y-%m-%d")} since:{fecha_desde.strftime("%Y-%m-%d")}',
                    'max_results': 100,
                    'count': 100,
                    'querySource': 'typed_query',
                    'product': 'Latest',
                    'min_faves': 100,
                    'cursor': cursor
                })
            else:
                break

        # Determino el sentimiento general y guardo variables utilies en el dataset.
        if tweets_utilizados > 0:
            overall_sentiment = get_overall_sentyment(total_compound_score, tweets_utilizados)
        else:
            overall_sentiment = 'neu'

        dataset.at[index, sentimiento_column] = overall_sentiment
        dataset.at[index, f'Cantidad_pos{"" if not column_prefix else "_"}{column_prefix if column_prefix else ""}'] = sentiments['pos']
        dataset.at[index, f'Cantidad_neg{"" if not column_prefix else "_"}{column_prefix if column_prefix else ""}'] = sentiments['neg']
        dataset.at[index, f'Cantidad_neu{"" if not column_prefix else "_"}{column_prefix if column_prefix else ""}'] = sentiments['neu']
        dataset.at[index, tweets_utilizados_column] = tweets_utilizados
        dataset.at[index, f'Compound total{"" if not column_prefix else " "}{column_prefix if column_prefix else ""}'] = total_compound_score
        dataset.to_csv(ruta_dataset, index=False, float_format='%.8f')

    return dataset
###################################################################################################

# Lógica para obtener el sentimiento específico de la moneda
def obtener_sentimiento_moneda(par, ruta_dataset):
    coin_related_terms = get_coin_related_terms(par)
    palabras_clave = " OR ".join(coin_related_terms)
    return obtener_sentimiento(ruta_dataset, palabras_clave, "coin")

###################################################################################################

# Lógica para obtener el sentimiento de individuos influyentes
def obtener_sentimiento_individuos(ruta_dataset):
    palabras_clave = "bitcoin OR btc OR cryptocurrency OR crypto OR CryptoNews"
    specific_authors = ['elonmusk', 'jack', 'VitalikButerin', 'cz_binance', 'aantonop', 'brian_armstrong']
    return obtener_sentimiento(ruta_dataset, palabras_clave, "referentes", specific_authors)

###################################################################################################

# Lógica para obtener el sentimiento de individuos influyentes
def obtener_whale_alerts_twitter(ruta_dataset):
    if not os.path.isfile(ruta_dataset):
        raise FileNotFoundError(f"El archivo {ruta_dataset} no existe.")

    dataset = pd.read_csv(ruta_dataset)

    tweets_utilizados_column = 'Tweets_Utilizados_whale_alert'

    if tweets_utilizados_column not in dataset.columns:
        dataset[tweets_utilizados_column] = 0
        
    headers = TWT_HEADERS

    for index, row in dataset.iterrows():
        
        if row[tweets_utilizados_column] > 0:
            continue

        fecha_desde = pd.to_datetime(row['Open_time'])
        fecha_hasta = fecha_desde + timedelta(days=1)

        print(f"Procesando fecha desde: {fecha_desde}, fecha hasta: {fecha_hasta}")

        params = {
            'variables': json.dumps({
                'rawQuery': f'(from:whale_alert) until:{fecha_hasta.strftime("%Y-%m-%d")} since:{fecha_desde.strftime("%Y-%m-%d")}',
                'max_results': 100,
                'count': 100,
                'querySource': 'typed_query',
                'product': 'Latest',
                'min_faves': 250
            }),
            'features': TWT_FEATURES,
        }

        tweets_utilizados = 0

        for page in range(20):  # Obtener un máximo de veinte páginas
            response = requests.get('https://twitter.com/i/api/graphql/ummoVKaeoT01eUyXutiSVQ/SearchTimeline',
                                    headers=headers, params=params)
            has_sleept, errored = process_response(response)

            if has_sleept:
                response = requests.get('https://twitter.com/i/api/graphql/ummoVKaeoT01eUyXutiSVQ/SearchTimeline',
                                        headers=headers, params=params)
            elif errored:
                continue

            response_json = response.json()
            tweets = response_json['data']['search_by_raw_query']['search_timeline']['timeline']['instructions'][0]

            if 'entries' in tweets:
                tweets = tweets['entries']
            else:
                tweets = []

            print(f"Tweets para la pagina {page}, total: {len(tweets)}")

            if len(tweets) == 0:
                break

            for tweet in tweets:
                tweet_text = get_tweet_text(tweet)

                if not len(tweet_text) > 0:
                    continue
                else:
                    # print(tweet_text)
                    tweets_utilizados += 1

            cursor = get_next_page_token(response_json)

            if not len(cursor) == 0:
                params['variables'] = json.dumps({
                    'rawQuery': f'(from:whale_alert) until:{fecha_hasta.strftime("%Y-%m-%d")} since:{fecha_desde.strftime("%Y-%m-%d")}',
                    'max_results': 100,
                    'count': 100,
                    'querySource': 'typed_query',
                    'product': 'Latest',
                    'min_faves': 100,
                    'cursor': cursor
                })
            else:
                break

        # Guardo la cantidad de tweets utilizados en el dataset
        dataset.at[index, tweets_utilizados_column] = tweets_utilizados
        dataset.to_csv(ruta_dataset, index=False, float_format='%.8f')

    return dataset
###################################################################################################

# Al ejecutar esta funcion sera necesario obtener un nuevo curl de la web y actualizar los headers.
# Lo que hago es copiar el dataset.csv a mano, renombrarlo y ejecutar la funcion sobre ese archivo,
# con el codigo que esta en los ejemplos de abajo.
## Con 5 paginas tarda aprox 19hs
## Con 10 paginas tarda aprox 37hs
def obtener_sentimiento_general(ruta_dataset):
    palabras_clave = "bitcoin OR cryptocurrency OR crypto OR CryptoNews"
    return obtener_sentimiento(ruta_dataset, palabras_clave, "")

###################################################################################################

# Lógica para obtener el precio de cierre, apertura, maximo y minimo de otras líderes como BTC, ETH y BNB
# Aprox 10hs
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

###################################################################################################

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

def obtener_aggregate_trades_coinbase(symbol, start_date, end_date):
    trades = []
    after_trade_time = None
    start_date = start_date.replace(tzinfo=timezone.utc)
    end_date = end_date.replace(tzinfo=timezone.utc)
    
    # print(start_date)
    # print(end_date)

    while True:
        request_path = f"/api/v3/brokerage/products/{symbol}/ticker"
        jwt_token = build_jwt_token(request_path)
        url = f"https://api.coinbase.com{request_path}"

        headers = {
            "Authorization": f"Bearer {jwt_token}",
        }

        params = {"limit": 1000, "start": int(start_date.timestamp()), "end": int(end_date.timestamp())}

        if after_trade_time:
            params["end"] = int(after_trade_time.timestamp())

        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            trades_data = response.json()
            # print(trades_data)
            if not trades_data or not trades_data['trades'] or len(trades_data['trades']) == 0:
                break
            
            trades.extend(trades_data['trades'])
            end_time_of_first_last_str = trades_data['trades'][-1]['time']
            end_time_of_last_trade = parse_datetime_string(end_time_of_first_last_str)
            after_trade_time = end_time_of_last_trade

            print(f"Siguiente limite a consultar: {after_trade_time}")
            # print(trades_data['trades'][-1]['trade_id'])

            if end_time_of_last_trade <= start_date:
                break
        elif response.status_code == 429: 
            print("Se ha alcanzado el límite de solicitudes. Esperando 10 segundos...")
            time.sleep(10)
        else:
            print("Error en la solicitud:", response.status_code)

    return trades


# Umbral equivale a cantidad de monedas
def obtener_whale_alerts_coinbase(start_date, end_date, moneda, umbral, precio_historico):
    fecha = start_date
    resultado = pd.DataFrame(columns=["Open_time", "buy_1000x_high_coinbase", "sell_1000x_high_coinbase", "total_trades_coinbase"])

    while fecha < end_date:
        trades = obtener_aggregate_trades_coinbase(moneda, fecha, fecha + timedelta(days=1))
        if trades and len(trades) > 0:
            precio_historico_filtered = precio_historico[precio_historico['Open_time'] == fecha]
            high_value = float(precio_historico_filtered['High'].max())
            
            print(f"Fecha sobre la que voy a calcular los whales: {fecha}, mayor precio para ese dia: {high_value}, umbral: {high_value * umbral}, total trades: {len(trades)}")
            
            buy_1000x_high_coinbase = sum(1 for trade in trades if trade['side'] == 'BUY' and float(trade['price']) * float(trade['size']) > high_value * umbral)
            sell_1000x_high_coinbase = sum(1 for trade in trades if trade['side'] == 'SELL' and float(trade['price']) * float(trade['size']) > high_value * umbral)

            resultado.loc[len(resultado)] = [pd.to_datetime(fecha), buy_1000x_high_coinbase, sell_1000x_high_coinbase, len(trades)]
            # print(trades)
            # print(resultado)

        fecha += timedelta(days=1)

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
coinbase_symbol = 'DOT-USD'
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

start_date = fecha_especifica - timedelta(days=(margin_days + wanted_previous_dates))
end_date = fecha_especifica + timedelta(days=1)

# obtener_historico_precio
# datos_candlestick = obtener_historico_precio(interval, start_time, end_time, binance_symbol)
# print(datos_candlestick)

# obtener_histórico_precio_influyentes
# datos_influentes = obtener_histórico_precio_influyentes(interval, start_time, end_time, ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
# print(datos_influentes)

# generar_dataset
# datos_completo = generar_dataset(interval, start_time, end_time, start_date, end_date, coinbase_symbol, binance_symbol, ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
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


# Calcular sentimiento general del mercado
# sentimiento_general = obtener_sentimiento_general("/Users/mmarchetta/Desktop/Tesis-2024/dataset_sentimiento_general_10_paginas.csv")

# Calcular sentimiento particular de la moneda
# sentimiento_moneda = obtener_sentimiento_moneda(binance_symbol, "/Users/mmarchetta/Desktop/Tesis-2024/dataset_sentimiento_moneda_10_paginas.csv")

# Calcular sentimiento de referentes de la industria
# sentimiento_referentes = obtener_sentimiento_individuos("/Users/mmarchetta/Desktop/Tesis-2024/dataset_sentimiento_referentes_10_paginas.csv")

# Calcular obtener_whale_alerts_twitter
# obtener_whale_alerts_twitter("/Users/mmarchetta/Desktop/Tesis-2024/dataset_whale_alerts_twt_10_paginas.csv")

# Calcular grandes transacciones de coinbase
historico_precio = obtener_historico_precio(interval, start_time, end_time, binance_symbol)
whale_alerts_coinbase = obtener_whale_alerts_coinbase(start_date, end_date, coinbase_symbol, 1000, historico_precio)
whale_alerts_coinbase['Open_time'] = pd.to_datetime(whale_alerts_coinbase['Open_time']).dt.date
guardar_dataset_en_csv(whale_alerts_coinbase, name="whale_alerts_coinbase.csv")
