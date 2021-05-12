
import numpy as np
import pandas as pd
from datetime import datetime


class CleanData:

    string_fields = ['home_team', 'away_team', 'stadium', 'won_toss', 'roof',
                     'surface', 'weather', 'fav_team', 'points_line_result']

    def __init__(self, data_frame):
        self.df = data_frame

    def strip_string_fields(self):
        for field in CleanData.string_fields:
            self.df[field] = self.df[field].astype(str)
            self.df[field] = self.df[field].str.strip()
        return

    def convert_time(self, col_name):
        self.df[col_name] = pd.to_datetime(self.df[col_name], format='00:%H:%M').dt.time
        return

    def convert_time_seconds(self, col_name):
        self.df[col_name] = pd.to_timedelta(self.df[col_name]).dt.total_seconds().astype(int)
        return

    def create_degree_field(self):
        degrees = self.df['weather'].str.extract(r'(^\d* degrees)')

        def degree_field(row):
            if not pd.isna(row[0]):
                return int(row[0].split(' ')[0])
            return 0

        d = degrees.apply(degree_field, axis=1)
        self.df['degrees'] = d
        return

    def create_wind_field(self):
        wind = self.df['weather'].str.extract(r'(wind \d* mph)')

        def wind_field(row):
            if not pd.isna(row[0]):
                return int(row[0].split(' ')[1])
            return 0

        w = wind.apply(wind_field, axis=1)
        self.df['wind'] = w
        return

    def create_humidity_field(self):
        humidity = self.df['weather'].str.extract(r'(relative humidity \d*%)')

        def humidity_field(row):
            if not pd.isna(row[0]):
                return int(row[0].split(' ')[-1][:-1])
            return 0

        h = humidity.apply(humidity_field, axis=1)
        self.df['humidity'] = h
        return

    def change_team_name(self, name, n_replace):
        self.df['home_team'] = self.df['home_team'].str.replace(name, n_replace)
        self.df['away_team'] = self.df['away_team'].str.replace(name, n_replace)
        self.df['fav_team'] = self.df['fav_team'].str.replace(name, n_replace)
        return

    def change_stadium_name(self, name, n_replace):
        self.df['stadium'] = self.df['stadium'].str.replace(name, n_replace)
        return

    def create_team_alias(self):
        self.df['home_team_alias'] = self.df['home_team'].str.split(' ').str[-1]
        self.df['away_team_alias'] = self.df['away_team'].str.split(' ').str[-1]
        return

    def convert_over_under(self):

        def over_under(row):
            if row['points_line_result'] == 'over':
                return 1
            return 0

        p = self.df.apply(over_under, axis=1)
        self.df['points_line_result'] = p
        return


if __name__ == '__main__':

    years = range(2003, 2021)
    weeks = range(1, 22)
    custom_date_parser = lambda x: datetime.strptime(x, '%d/%m/%Y')

    for year in years:
        for week in weeks:
            df = pd.read_csv('./data/scraped_data/' + str(year) + '/week_' + str(week) + '.csv',
                             parse_dates=['game_date'],
                             date_parser=custom_date_parser)

            df.reset_index(drop=False, inplace=True)

            c = CleanData(df)

            # create year column
            df['year'] = year

            # create week column
            df['week'] = week

            # strip string fields
            c.strip_string_fields()

            # convert duration column
            c.convert_time('duration')

            # convert duration, home, away time of possession
            c.convert_time_seconds('home_time_possession')
            c.convert_time_seconds('away_time_possession')

            # create degree field
            c.create_degree_field()

            # create wind field
            c.create_wind_field()

            # create humidity field
            c.create_humidity_field()

            # change team name
            c.change_team_name('Washington Football Team', 'Washington Redskins')

            # change stadium name

            # arizona
            c.change_stadium_name('University of Phoenix Stadium', 'State Farm Stadium')

            # raiders
            c.change_stadium_name('Ring Central Coliseum', 'Oakland Coliseum')
            c.change_stadium_name('McAfee Coliseum', 'Oakland Coliseum')
            c.change_stadium_name('Network Associates Coliseum', 'Oakland Coliseum')
            c.change_stadium_name('O.co Coliseum', 'Oakland Coliseum')

            # 49ers
            c.change_stadium_name('Candlestick Park', 'Candlestick Park, San Francisco')
            c.change_stadium_name('3Com Park', 'Candlestick Park, San Francisco')
            c.change_stadium_name('Monster Park', 'Candlestick Park, San Francisco')

            # vikings
            c.change_stadium_name('Mall of America Field', 'U.S. Bank Stadium')
            c.change_stadium_name('Hubert H. Humphrey Metrodome', 'U.S. Bank Stadium')

            # chargers
            c.change_stadium_name('StubHub Center', 'Dignity Health Sports Park')

            # browns
            c.change_stadium_name('Cleveland Browns Stadium', 'FirstEnergy Stadium')

            # broncos
            c.change_stadium_name('Sports Authority Field at Mile High', 'Empower Field at Mile High')

            # jags
            c.change_stadium_name('EverBank Field', 'TIAA Bank Stadium')
            c.change_stadium_name('Alltel Stadium', 'TIAA Bank Stadium')

            # rams
            c.change_stadium_name('Edward Jones Dome', "The Dome at America's Center")

            # titans
            c.change_stadium_name('The Coliseum', 'Nissan Stadium')
            c.change_stadium_name('LP Field', 'Nissan Stadium')

            # giants/jets
            c.change_stadium_name('New Meadowlands Stadium', 'MetLife Stadium')
            c.change_stadium_name('Giants Stadium', 'MetLife Stadium')

            # panthers
            c.change_stadium_name('Ericsson Stadium', 'Bank of America Stadium')

            # miami
            c.change_stadium_name('Pro Player Stadium', 'Hard Rock Stadium')
            c.change_stadium_name('Dolphin Stadium', 'Hard Rock Stadium')

            # seahawks
            c.change_stadium_name('Seahawks Stadium', 'CenturyLink Field')

            # colts
            c.change_stadium_name('RCA Dome', 'Lucas Oil Stadium')

            # saints
            c.change_stadium_name('Louisiana Superdome', 'Mercedes-Benz Superdome')

            # create alias for teams EverBank Field The Dome at America's Center RCA Dome
            c.create_team_alias()

            # create points line result
            c.convert_over_under()

            df = df[['game_date', 'year', 'week', 'home_team_alias', 'away_team_alias', 'home_score', 'away_score',
                     'stadium', 'degrees', 'wind', 'humidity', 'roof', 'fav_team', 'handicap', 'points_line',
                     'points_line_result', 'home_first_downs', 'away_first_downs', 'home_num_rushes', 'home_rush_yards',
                     'home_rush_tds', 'away_num_rushes', 'away_rush_yards', 'away_rush_tds', 'home_pass_comp',
                     'home_pass_att', 'home_pass_yards', 'home_pass_tds', 'home_pass_ints', 'away_pass_comp',
                     'away_pass_att', 'away_pass_yards', 'away_pass_tds', 'away_pass_ints', 'home_sacks_allowed',
                     'home_sacks_yards_allowed', 'away_sacks_allowed', 'away_sacks_yards_allowed',
                     'home_net_pass_yards', 'away_net_pass_yards', 'home_total_yards', 'away_total_yards',
                     'home_fumbles', 'home_fumbles_lost', 'away_fumbles', 'away_fumbles_lost', 'home_turnovers',
                     'away_turnovers', 'home_num_penallity', 'home_penallity_yards', 'away_num_penallity',
                     'away_penallity_yards', 'home_third_downs', 'home_third_downs_converted', 'away_third_downs',
                     'away_third_downs_converted', 'home_fourth_downs', 'home_fourth_downs_converted',
                     'away_fourth_downs', 'away_fourth_downs_converted', 'home_time_possession', 'away_time_possession']]

            # save to new file location
            df.to_csv('./data/cleaned_data/' + str(year) + '/week_' + str(week) + '.csv', index=False)