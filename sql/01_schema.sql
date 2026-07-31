CREATE DATABASE retail_forecasting;
USE retail_forecasting;

CREATE TABLE stores (
    store_id INT PRIMARY KEY,
    store_type VARCHAR(5),
    assortment VARCHAR(5),
    competition_distance INT,
    competition_open_since_month INT,
    competition_open_since_year INT,
    promo2 TINYINT,
    promo2_since_week INT,
    promo2_since_year INT,
    promo_interval VARCHAR(50)
);

CREATE TABLE sales (
    sale_id INT AUTO_INCREMENT PRIMARY KEY,
    store_id INT,
    sale_date DATE,
    day_of_week INT,
    sales_amount INT,
    customers INT,
    is_open TINYINT,
    promo TINYINT,
    state_holiday VARCHAR(5),
    school_holiday TINYINT,
    FOREIGN KEY (store_id) REFERENCES stores(store_id)
);
LOAD DATA LOCAL INFILE '/Users/karansood/Desktop/rossmann-store-sales/store.csv'
INTO TABLE stores
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(store_id, store_type, assortment, competition_distance, @comp_month, @comp_year, promo2, @promo2_week, @promo2_year, promo_interval)
SET
  competition_open_since_month = NULLIF(@comp_month, ''),
  competition_open_since_year  = NULLIF(@comp_year, ''),
  promo2_since_week            = NULLIF(@promo2_week, ''),
  promo2_since_year            = NULLIF(@promo2_year, '');

 LOAD DATA LOCAL INFILE '/Users/karansood/Desktop/rossmann-store-sales/train.csv'
INTO TABLE sales
FIELDS TERMINATED BY ',' ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(store_id, day_of_week, sale_date, sales_amount, customers, is_open, promo, state_holiday, school_holiday);

SELECT s.store_id, s.store_type, sa.sale_date, sa.sales_amount
FROM sales sa
JOIN stores s ON sa.store_id = s.store_id
LIMIT 10;


