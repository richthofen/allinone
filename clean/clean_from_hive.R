#library("quantmod")
library(SparkR)
# 这个sc 要自己init ，sparkR she’ll里会把这个自动init掉
sc <- sparkR.init()
hiveContext <- sparkRHive.init(sc)
results <- sql(hiveContext, "select * from  dw.detail1   where symbol='sh600000' limit 100")
head(results)
