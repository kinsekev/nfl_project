import json
import numpy as np
import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.distance import great_circle
from geopy.geocoders import Nominatim
from pprint import pprint
from config import HOME_TEAM_ALIAS, STADIUM


geolocator = Nominatim(user_agent="nfl_app")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)


def distance_stadiums(team_stadium):
    search = team_stadium.values()
    locations = [geocode(s) for s in search]

    stadium_locations = dict(zip(team_stadium.keys(), locations))
    # return stadium_locations

    stadium_coords = {k: (v.latitude, v.longitude) for k, v in stadium_locations.items()}

    outside_dct = {}
    for team1 in stadium_coords.keys():
        loc1 = stadium_coords[team1]
        inside_dct = {}
        for team2 in stadium_coords.keys():
            loc2 = stadium_coords[team2]
            distance = round(great_circle(loc1, loc2).miles)
            inside_dct[team2] = distance
        outside_dct[team1] = inside_dct
    return outside_dct


if __name__ == '__main__':

    years = range(2003, 2021)
    weeks = range(1, 18)
    stadium_years = {}

    for year in years:
        games = [pd.read_csv('./data/cleaned_data/' + str(year) + '/week_' + str(i) + '.csv') for i in weeks]
        games = pd.concat(games)
        team_stadium = list(zip(games[HOME_TEAM_ALIAS], games[STADIUM]))

        dct = {}
        for team in team_stadium:
            if team in dct:
                dct[team] += 1
            else:
                dct[team] = 1

        dct = {k: v for k, v in dct.items() if v > 3}

        pprint(dct)

        team_stadium_dct = {}
        for team, stadium in dct.keys():
            team_stadium_dct[team] = stadium

        s = distance_stadiums(team_stadium_dct)

        stadium_years[year] = s
        print(year)

    pprint(stadium_years)

    with open('./data/distance.json', 'w') as fp:
        json.dump(stadium_years, fp)

