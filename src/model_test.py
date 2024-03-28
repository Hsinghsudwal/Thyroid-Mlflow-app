import pandas as pd
import joblib

svc=joblib.load('models/model.joblib')
df=pd.read_csv('artifacts/transformer.csv')#,index_col=0)

target=df.drop(['classes'],axis=1)

testing=target.iloc[400].values.reshape(1,-1)
raw=df['classes']
print(raw.iloc[400])

test=svc.predict(testing)
print(test)

if test==0:
    print('Compensated Hypothyroid')
elif test==1:
    print('Negative')
elif test==2:
    print('Primary Hypothyroid')
else:
    print('Secondary Hypothyroid')
