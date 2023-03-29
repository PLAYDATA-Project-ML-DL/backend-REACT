import pandas as pd
import pymysql
from sqlalchemy import create_engine

df = pd.read_csv('D:/db/DB_DATA(0329).csv', index_col=0)#

db = pymysql.connect(user = 'root', host = 'localhost', passwd='root', port=3306, db='django')

engine = create_engine("mysql+pymysql://root:"+"root"+"@localhost:3306/django?charset=utf8")
conn = engine.connect()
df.to_sql(name = "medicines_medicine", if_exists='append', con=engine, index=False, index_label="id")#if_exists='replace',
print("input medicines DB complete")
conn.close()

# admin 에서 같이 활용하려면 제공하는 속성과 column의 일체화가 필요하다. 