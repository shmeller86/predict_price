from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from src.Load import LoadBinanceHistoryData


# market='SPOT'
# sym = 'BTCUSDT'
# tf = '1h'
# f = '2022-01-01 00:00:00'
# t = '2022-02-28 23:59:59'
# lb = LoadBinanceHistoryData(market, sym, tf, f, t)
# lb.load()

df = pd.read_csv('S_BTCUSDT_1h__20220101_0000__20220228_2359_.csv')
df = df.rename(columns={
    "Open time": "ts",
    "Open": "o",
    "High": "h",
    "Low": "l",
    "Close": "c",
    "Volume": "v",
    "Quote asset volume": "qav",
    "Number of trades": "not",
    "Taker buy base asset volume": "tbbav",
    "Taker buy quote asset volume": "tbqav"
})

df = df.loc[len(df)-1000:]

df["ts"] = [datetime.fromtimestamp(x) for x in df.ts]
df = df.set_index('ts')
df.drop(columns=['qav','tbbav','tbqav', 'not'], axis=1, inplace=True)

df['ema_f'] = df['c'].ewm(span=5, adjust=False).mean()
df['ema_m'] = df['c'].ewm(span=8, adjust=False).mean()
df['ema_l'] = df['c'].ewm(span=13, adjust=False).mean()

df['ema_f_m__bin'] = (df['ema_f'] >= df['ema_m']).astype(int)
df['ema_m_l__bin'] = (df['ema_m'] >= df['ema_l']).astype(int)
df['ema_f_l__bin'] = (df['ema_f'] >= df['ema_l']).astype(int)

df['p_chg'] = (df.c - df.o) / df.o *100
df['p_cdl_full_size'] = (df.h - df.l) / df.l * 100

df = df.assign(p_cdl_bottom_fitil=(abs((df.c - df.l) / df.l * 100)).where(df.o >= df.c, abs((df.o - df.l) / df.l * 100)))
df = df.assign(p_cdl_top_fitil=(abs((df.h - df.o) / df.o * 100)).where(df.o >= df.c, abs((df.h - df.c) / df.c * 100)))
df = df.assign(p_cdl_body=(abs((df.o - df.c) / df.c * 100)).where(df.o >= df.c, abs((df.c - df.o) / df.o * 100)))

percent = 2.0
df['diff_c3_buy'] = (df.h.shift(-2) - df.c) / df.c * 100
df['diff_c3_sell'] = (df.c - df.l.shift(-2)) / df.l * 100
df['diff_c3_buy_b'] = (((df.h.shift(-2) - df.c) / df.c * 100) > percent).astype(int)
df['diff_c3_sell_b'] = (((df.c - df.l.shift(-2)) / df.l * 100) > percent).astype(int)
df = df[:-2]
y = np.array(list(zip(df['diff_c3_buy_b'].astype(int), df['diff_c3_sell_b'].astype(int))))
df_copy = df.copy()
df = df.drop(columns=['diff_c3_buy', 'diff_c3_sell', 'diff_c3_buy_b', 'diff_c3_sell_b'], axis=1)

np.set_printoptions(suppress=True)
df.reset_index(drop=True, inplace=True)
scaller = MinMaxScaler(feature_range=(-1,1))
df = np.array(df)
df = scaller.fit_transform(df)
X = np.around(df, decimals=3)

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y,test_size=.2,random_state=0)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtrain, ytrain)
dump(knn, 's_btc_1h_2percent.joblib') 

classConf = knn.score(Xtest, ytest)
print("KNeighborsClassifier confidence: ",classConf)
p = knn.predict(X)
df = df_copy
df['p_buy'], df['p_sell'] = p.T

print("Всего сигналов в LONG " +  str((df.p_buy == 1).sum()))
print("Всего сигналов в SHORT " +  str((df.p_sell == 1).sum()))
print("Размерность массива X_train: {}".format(Xtrain.shape)) 
print("Размерность массива у_train: {}".format(ytrain.shape))
print("Размерность массива Х test: {}".format(Xtest.shape)) 
print("Размерность массива y_test: {}".format(ytest.shape))

df.loc[df.p_buy == 1, 'predict_ii_buy'] = 1 * df.c + (df.c*0.001)
df.loc[df.p_sell == 1, 'predict_ii_sell'] = 1 * df.c - (df.c*0.001)
df.loc[df.diff_c3_buy_b == 1, 'predict_buy'] = 1 * df.c
df.loc[df.diff_c3_sell_b == 1, 'predict_sell'] = 1 * df.c

plt.style.use('fivethirtyeight')
fig = plt.figure(figsize=(24.,6.))
ax = fig.add_subplot(1,1,1)
count = 800
limit = -1
plt.plot(df.predict_buy[count:limit], color='g', marker='o', ms=25.,alpha=.1)
plt.plot(df.predict_sell[count:limit], color='r', marker='o', ms=25.,alpha=.1)
for t, c, b, s , sigb, sigs in zip(df.index.values[count:limit], df.c[count:limit], df.diff_c3_buy[count:limit],df.diff_c3_sell[count:limit], df.predict_ii_buy[count:limit], df.predict_ii_sell[count:limit]):
    if sigb > 0:
        plt.text(x=t, y=c-(c*0.005), s=str(b)[:4], horizontalalignment='center', color='darkgreen')
    if sigs > 0:
        plt.text(x=t, y=c+(c*0.005), s=str(s)[:4], horizontalalignment='center', color='red')

plt.plot(df.predict_ii_buy[count:limit], color='g', marker='P', ms=10.) 
plt.plot(df.predict_ii_sell[count:limit], color='r', marker='P', ms=10.)
plt.plot(df.c[count:limit], color='black', linewidth=2)
plt.title('PREDICT', fontsize=24)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price', fontsize=18)
plt.show()

