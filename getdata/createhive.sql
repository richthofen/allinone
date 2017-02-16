CREATE TABLE `dw.detail`(
  `time` timestamp,
  `price` float,
  `float_rate` float,
  `vol` int,
  `turnover` int,
  `op` string)
PARTITIONED BY (
  `symbol` string,
  `date_time` date)
ROW FORMAT DELIMITED
  FIELDS TERMINATED BY ',';
