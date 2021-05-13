
# American Football 

Goal is to use American Football data to predict if the home team will win a given game.

Data is scraped off of website - https://www.pro-football-reference.com/

In this project, features are engineered using source data from https://www.pro-football-reference.com/ such as: divisonal games, enclosed roof games, how long since a team previous played, how far has a team travelled to an away game.

Further, the PageRank[1] algorithm is used to weight the link between two football teams in order to rank each team. The features used to rank teams were based on margin of victory, passing yards scored/conceded, rushing yards scored/conceded, turnovers scored/conceded, sacks scored/conceded, penalty yards given/conceded, 
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

This script scrapes the data off https://www.pro-football-reference.com/ in order to extract source data from the web page for the project. 
The year and week can be modified to pull the data required. 

```
Run cells in the iPython notebook: Scrape NFL Data - Pro Football Reference.ipynb
```

Data will be stored in the data/scraped_data folder

## Clean data set

The clean_data.py script cleans the scraped data by stripping the string fields, converting time to seconds, spliting the weather field, changing team names to team alias names and changes stadium names that were unable to be located by the geolocator.

To run the code:
```
python clean_data.py
```

The original data sets must be in the data/scraped_data folder. The new file will also populate into the data/cleaned_data folder.

## Create distance between stadiums JSON

The stadium_distances.py script creates a JSON object to find the distances between every stadium for each league season(to accommodate teams moving stadiums). 

To run the code:
```
python stadium_distances.py
```

The JSON object is named as distance.json and placed into the data/ folder. 

## Enrich the data set

The enrich_data.py script, enriches the cleaned data set in order to generate features that can be used for modelling. The features that are created are divisonal games, enclosed roof games, how far an away team has travelled, how long since a teams last game was played. Further, the PageRank algorithim, separately generates features for margin of victory, passing yards, rushing yards, turnovers, sacks, penalty yards, 3rd/4th down conversions and time of possession to rank each team based on these features.

To run the code:
```
python enrich_data.py
```

The original data sets must be in the data/clean_data folder. The new file will also populate into the data/enriched_data folder.


## Predictions

The PCA with Random Forest Predictions.ipynb notebook utilizes the enriched data set to make predictions for a home team victory. The model used is a combination of PCA with Random Forest with the hyper parameters tuned across 5-fold cross validation. The test set score produces an accuracy of 68%.

```
Run cells in the iPython notebook: PCA with Random Forest Predictions.ipynb
```

# References

[1] An application of Google's PageRank to NFL Rankings  https://projecteuclid.org/download/pdf_1/euclid.involve/1513733537
