import sqlite3
import numpy
from os import path
_MODULE_PATH = path.dirname(__file__)

conn = sqlite3.connect("sqlite.db")  #创建sqlite.db数据库
# print ("open database success")
# # conn.execute("drop table IF EXISTS student")
# query = """create table IF NOT EXISTS student1(
#     customer VARCHAR(20),
#     produce VARCHAR(40),
#     amount FLOAT,
#     date DATE   
# );"""
# conn.execute(query)
# print ("Table created successfully")

#在表中插入数据

''' 方法1 '''
#data = '''INSERT INTO student(customer,produce,amount,date)\
#    VALUES("zhangsan","notepad",999,"2017-01-02")'''
#conn.execute(data)
#data = '''INSERT INTO student(customer,produce,amount,date)\
#    VALUES("lishi","binder",3.45,"2017-04-05")'''
#conn.execute(data)
#conn.commit()

''' 方法2 '''
# statement = "INSERT INTO student VALUES(?,?,?,?)"
# data = [("zhangsan1","notepad",999,"2017-01-02"),("lishi1","binder",3.45,"2017-04-05")]
# conn.executemany(statement, data)
# conn.commit()

# curson = conn.execute("select * from student")
# conn.commit()
# print (curson)
# rows = curson.fetchall()
# print (rows)
# conn.close()
conn = sqlite3.connect("sqlite.db")  #创建sqlite.db数据库
def initDB(symbols = []):
    for symbol in symbols :
        create_data_sql = """create table IF NOT EXISTS data%s (
            date DATE PRIMARY Key,
            open REAL,
            close REAL,
            high REAL,
            low REAL,
            volume REAL,
            best    REAL DEFAULT 0
        );""" % symbol
        print (create_data_sql)
        conn.execute(create_data_sql)
        print ("create table %s success " % symbol)
        ## 通一个date 同一次预测
        create_predict_sql = """ create table IF NOT EXISTS predict%s (
            date DATE,
            open REAL,
            close REAL,
            high REAL,
            low REAL,
            volume REAL,
            best    REAL DEFAULT 0 
        ) """ % symbol
        conn.execute(create_predict_sql)
        # status 0. holding,
        #  1. T, T-upper, T-loss
        #  2. failed 
        #  3. pre 预测
        #  4. sold
        #  5. selling
        
    create_hold_sql = """create table IF NOT EXISTS hold (
        symbol VARCHAR(6),
        date DATE,
        status VARCHAR(20),
        checkin REAL,
        checkout REAL,
        profit REAL,
        original VARCHAR(20),
        indate DATE,
        outdate DATE,
        primary key (symbol, date)
    )
    """
    # print (create_hold_sql)
    conn.execute(create_hold_sql)
    conn.commit()
    conn.close()
def get_data(symbol):
    c = conn.cursor()
    statement = """select * from data%s order by date""" % symbol
    ret = c.execute(statement).fetchall()
    ret = numpy.array(ret)
    return ret
def get_data_from_date(symbol, dt):
    c = conn.cursor()
    statement = """select * from data%s where date>='%s' order by date desc""" % (symbol, dt)
    ret = c.execute(statement).fetchall()
    ret = numpy.array(ret)
    return ret
def get_data_lastest(symbol):
    c = conn.cursor()
    # statement = """select * from data%s  order by date limit -1, 10""" % (symbol)
    statement = """select * from ( select * from data%s  order by date desc  limit 6 ) order by date""" % (symbol)
    ret = c.execute(statement).fetchall()
    ret = numpy.array(ret)
    return ret
def get_predict(symbol, dt):
    c = conn.cursor()
    statement = """select * from predict%s where date='%s'""" % (symbol, dt)
    ret = c.execute(statement).fetchall()
    ret = numpy.array(ret)
    return ret
def get_predict_from_date(symbol, dt):
    c = conn.cursor()
    statement = """select * from predict%s where date='%s' """ % (symbol, dt)
    # print(statement)
    ret = c.execute(statement).fetchall()
    ret = numpy.array(ret)
    return ret
def get_latest_predict(symbol):
    c = conn.cursor()
    statement = """select * from (select * from predict%s order by date desc limit 10) order by date""" % (symbol)
    ret = c.execute(statement).fetchall()
    ret = numpy.array(ret)
    return ret

def add_tran_data(data, symbol):
    conn = sqlite3.connect("sqlite.db")  #创建sqlite.db数据库
    statement = "delete from data%s ;" % symbol
    conn.execute(statement)
    statement = """INSERT or ignore INTO data%s VALUES(?,?,?,?,?,?,?)""" % symbol
    conn.executemany(statement, data)
    conn.commit()
def add_predict(data, symbol, dt):
    # conn = sqlite3.connect("sqlite.db")  #创建sqlite.db数据库
    dts = numpy.array([dt]*10).reshape(10, 1)
    data = numpy.append(dts, data, axis=1)
    # data = data
    print (data)
    statement = """INSERT or ignore INTO predict%s (date, open, close,high, low, volume, best)  VALUES(?,?,?,?,?,?,?)""" % symbol
    conn.executemany(statement, data)
    conn.commit()
def add_data(data, symbol=""):
    # conn = sqlite3.connect("sqlite.db")  #创建sqlite.db数据库
    if symbol == "":
        symbol = data[0][6]
    data = data[:,[0,1,2,3,4,5]]
    statement = """INSERT or ignore INTO data%s (date, open, close,high, low, volume)  VALUES(?,?,?,?,?,?);
                    """ % symbol
    print(statement)
    conn.executemany(statement, data)
    conn.commit()

    # c = conn.cursor()
    # adj_date =  """update data%s set date=substr (date,0,5) || "-" || substr(date,5,2) ||"-" || substr(date, 7,2);
    #                 """ % symbol
    # c.execute(adj_date)

def get_hold_data(status):
    c = conn.cursor()
    statement = """select * from hold where status='%s'""" % status
    print(statement)
    ret = c.execute(statement).fetchall()
    ret = numpy.array(ret)
    return ret
def get_hold_holding_data(symbol):
    c = conn.cursor()
    statement = """select * from hold where status='%s' and symbol='%s'""" % (str("holding"), symbol)
    print(statement)
    ret = c.execute(statement).fetchall()
    ret = numpy.array(ret)
    return ret
def get_hold_pre_data():
    return get_hold_data( "pre")
def get_hold_all_holding_data():
    return get_hold_data( "holding")
def get_holding():
    c = conn.cursor()
    statement = """select * from hold where status='%s'""" % str("holding")
    ret = c.execute(statement).fetchall()
    ret = numpy.array(ret)
    return ret
def add_hold(data):
    # statement = """INSERT or ignore INTO hold (symbol, date, status, checkin, checkout, profit, original, indate, outdate)  VALUES(?,?,?,?,?,?)""" 
    statement = """INSERT or ignore INTO hold (symbol, date, status, checkin, checkout, profit, original, indate, outdate)  VALUES(?,?,?,?,?,?,?,?,?)""" 
    conn.executemany(statement, data)
    conn.commit()
def update_hold_end(status, checkout, profit, outdate, dt, symbol):
    statement = """update hold set status='%s', checkout=%f, profit=%f, outdate='%s'  where date = '%s' and symbol='%s'"""  % (status, checkout, profit, outdate, dt, symbol)
    print (statement)
    conn.execute(statement)
    conn.commit()
def update_hold_data( dt, symbol, status="pre", checkout=0, profit=0, outdate=""):
    statement = """update hold set status='%s', checkout=%f, profit=%f, outdate='%s'  where date = '%s' and symbol='%s'"""  % (status, checkout, profit, outdate, dt, symbol)
    print (statement)
    conn.execute(statement)
    conn.commit()


if __name__ == "__main__":
    abs_path =  path.join(_MODULE_PATH, 'back')
    with open(abs_path) as file:
    # with open('back') as file:
        data = file.read().split("\n")
        initDB(data)