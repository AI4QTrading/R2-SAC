import pandas as pd

data = pd.read_pickle('/data/hxyz/mindgo/zz1000.pkl')

df=data.groupby('code').apply(lambda x:x['close'].rolling(window=7).std())
df=df.reset_index()
df=df.drop(columns=['level_1'])
df=df.dropna()
x=[i for i in range(len(data.date.unique())-6)]
y=[x for _ in range(705)]
z=sum(y,[])
df['time']=z
df_event=df.sort_values(['code','close'])
dfx=df_event.groupby('code').apply(lambda x:x.iloc[-100:])
dfx=dfx.reset_index(drop=True)
id=[i for _ in range(100) for i in range(705)]
dfx['id']=id
dfx=dfx.rename(columns={'code':'event','close':'option1'})
order=['id','time','event','option1']
dfx=dfx[order]
dfx.to_csv('./zz1000_100.csv')
