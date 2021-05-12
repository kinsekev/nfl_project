
# American Football 

Goal is to use American Football data to predict if the home team will win a given game.

Data is scraped off of website - https://www.pro-football-reference.com/

In this project, features are engineered using source data from https://www.pro-football-reference.com/ such as: divisonal games, enclosed roof games, how long since a team previous played, 
how far has a team travelled to an away game.

Further, the PageRank algorithm is used to rank teams based on margin of victory, passing yards scored/conceded, rushing yards scored/conceded, turnovers scored/conceded, sacks scored/conceded, penalty yards given/conceded, 
3rd/4th down conversion rates given/conceded and time of possession for and against.


# Requirements

Install the required packages below

```
conda install numpy
conda install pandas
conda install matplotlib
conda install scikit-learn
conda install geopy
```

# Running the code

## Scrape data set

```
Run cells in the iPython notebook: Scrape NFL Data - Pro Football Reference.ipynb
```

Can modify the year and weeks in which to scrape.

Data will be stored in the data/scraped_data folder

## Clean data set

To run the code:
```
python clean_data.py
```

The original data sets must be in the data/scraped_data folder. The new file will also populate into the data/cleaned_data folder.

## Create distance between stadiums matrix

To run the code:
```
python stadium_distances.py
```

A JSON object will be created and placed into the data/ folder. 

## Enrich the data set

To run the code:
```
python enrich_data.py
```

The original data sets must be in the data/clean_data folder. The new file will also populate into the data/enriched_data folder.


## Predictions

```
Run cells in the iPython notebook: PCA with Random Forest Predictions.ipynb
```

# References

An application of Google's PageRank to NFL Rankings  https://projecteuclid.org/download/pdf_1/euclid.involve/1513733537