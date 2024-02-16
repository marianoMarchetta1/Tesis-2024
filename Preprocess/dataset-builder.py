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
    histórico_precio_influyentes = obtener_histórico_precio_influyentes(interval, start_time, end_time, monedas_lideres)
    indicadores_tecnicos = calcular_indicadores_tecnicos(histórico_precio)
    whale_alerts_binance = obtener_whale_alerts_binance(start_time, end_time, moneda, 100000, histórico_precio)
    sentimiento_general = obtener_sentimiento_general()
    sentimiento_moneda = obtener_sentimiento_moneda(moneda)
    sentimiento_individuos = obtener_sentimiento_individuos()
    
    guardar_dataset_en_csv(histórico_precio, name="histórico_precio.csv")
    guardar_dataset_en_csv(histórico_precio_influyentes, name="histórico_precio_influyentes.csv")
    guardar_dataset_en_csv(indicadores_tecnicos, name="indicadores_tecnicos.csv")
    guardar_dataset_en_csv(whale_alerts_binance, name="whale_alerts_binance.csv")
    
    dataset = pd.merge(histórico_precio, histórico_precio_influyentes, on='Open_time', how='outer')
    dataset = pd.merge(dataset, indicadores_tecnicos, on='Open_time', how='outer')
    dataset = pd.merge(dataset, whale_alerts_binance, on='Open_time', how='outer')
    
    dataset = dataset[39:] # Elimino los primeros 39 días para evitar valores NaN en los indicadores técnicos

    guardar_dataset_en_csv(dataset)

    return dataset

def obtener_sentimiento_general():
    # Lógica para obtener el sentimiento general del mercado
    # Retorna un diccionario {fecha: sentimiento}
    return

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
            
            print(fecha_formato)
            print(precio_historico['Open_time'].dt.date)
            
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

utc_timezone = pytz.timezone('UTC')
binance_symbol = 'DOTUSDT'
symbol = 'btc'
interval = '1d'  # Obtener datos diarios
# Retroceder 2 años desde la fecha específica y convertir a milisegundos
# A mi me interesa tomar 730 dias (2 años), a eso le agrego 40 dias mas,
# porque algunos indicadores tecnicos necesitan hasta 33 dias previos para ser calculados,
# sino me sucede que los primeros dias de la serie tienen ciertos indicadores en NaN.
# A su vez, le agrego un dia mas a la resta por el formato UTC de la API de binance
margin_days = 0 # 40 días de margen para los indicadores tecnicos
wanted_previous_dates = 0 # 2 años
start_time = int((fecha_especifica - timedelta(days=(margin_days + wanted_previous_dates + 1))).timestamp() * 1000)
end_time = int((fecha_especifica + timedelta(days=(1))).timestamp() * 1000)

# obtener_historico_precio
# datos_candlestick = obtener_historico_precio(interval, start_time, end_time, binance_symbol)
# print(datos_candlestick)

# obtener_histórico_precio_influyentes
# datos_influentes = obtener_histórico_precio_influyentes(interval, start_time, end_time, ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
# print(datos_influentes)

# generar_dataset
# datos_completo = generar_dataset(interval, start_time, end_time, binance_symbol, symbol, ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
# print(datos_completo)

# calcular_indicadores_tecnicos
# datos_candlestick = obtener_historico_precio(interval, start_time, end_time, binance_symbol)
# indicadores_tecnicos = calcular_indicadores_tecnicos(datos_candlestick)
# print(indicadores_tecnicos)


# Calcular aggregated_trades
# aggregated_trades = obtener_aggregate_trades(binance_symbol, start_time, end_time)
# print(aggregated_trades)

# Calcular obtener_whale_alerts_binance
historico_precio = obtener_historico_precio(interval, start_time, end_time, binance_symbol)
print(historico_precio)
# print(datetime.utcfromtimestamp(start_time / 1000))
# print(datetime.utcfromtimestamp(end_time / 1000))
whale_alerts = obtener_whale_alerts_binance(start_time, end_time, binance_symbol, 1000, historico_precio)
print(whale_alerts)
