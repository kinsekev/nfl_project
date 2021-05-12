
import json
import pandas as pd
import numpy as np
from pagerank_factory import hmatrix_factory
from config import AFC_EAST, AFC_NORTH, AFC_SOUTH, AFC_WEST, NFC_EAST, NFC_NORTH, NFC_SOUTH, NFC_WEST, HOME_TEAM_COL, \
                    AWAY_TEAM_COL, GAME_DATE_COL, IS_DIVISIONAL_GAME, IS_DOME_GAME, AWAY_DISTANCE_TRAVELLED, YEAR, \
                        HOME_PREVIOUS_GAME, AWAY_PREVIOUS_GAME, HOME_SCORE, AWAY_SCORE, HOME_PASS_YARDS, AWAY_PASS_YARDS, \
                            HOME_RUSH_YARDS, AWAY_RUSH_YARDS, HOME_TURNOVERS, AWAY_TURNOVERS, HOME_SACKS_ALLOWED, \
                                AWAY_SACKS_ALLOWED, HOME_PENALTY, AWAY_PENALTY, HOME_THIRD_DOWNS_CONVERTED, \
                                    AWAY_THIRD_DOWNS_CONVERTED, HOME_FOURTH_DOWNS_CONVERTED, AWAY_FOURTH_DOWNS_CONVERTED, \
                                        HOME_TIME_OF_POSSESSION, AWAY_TIME_OF_POSSESSION, HOME_TEAM_WIN


class EnrichData(object):

    DIVISIONS = [AFC_EAST, AFC_NORTH, AFC_SOUTH, AFC_WEST, NFC_EAST, NFC_NORTH, NFC_SOUTH, NFC_WEST]

    def __init__(self, files):
        self.files = files
        self.file_path = './data/cleaned_data/'
        self.cur_year = int(files[0][GAME_DATE_COL].dt.year[0])
        self.end_week = 17
        with open('distance.json', 'r') as fp:
            self.distance = json.load(fp)

    def create_divisional_game(self):

        def is_divisional(row, divs):
            home_team, away_team = row[HOME_TEAM_COL], row[AWAY_TEAM_COL]
            for div in divs:
                if home_team in div:
                    if away_team in div:
                        return 1
            return 0

        for f in self.files:
            is_div = f.apply(is_divisional, args=(EnrichData.DIVISIONS,), axis=1)
            f[IS_DIVISIONAL_GAME] = is_div
        return

    def create_dome_game(self):

        def is_dome_game(row):
            if row['roof'] in ['dome', 'retractable roof (closed)']:
                return 1
            return 0

        for f in self.files:
            is_dome = f.apply(is_dome_game, axis=1)
            f[IS_DOME_GAME] = is_dome
        return

    def create_away_distance_travelled(self):

        def distance_travelled(row, distance):
            h_team, a_team, year = row[HOME_TEAM_COL], row[AWAY_TEAM_COL], str(row[YEAR])
            if distance[year][a_team][h_team]:
                return distance[year][a_team][h_team]
            return None

        for f in self.files:
            away_dist = f.apply(distance_travelled, args=(self.distance,), axis=1)
            f[AWAY_DISTANCE_TRAVELLED] = away_dist
        return

    def create_previous_game_played(self):
        # create dict for date of last game played
        dct = {}
        t = pd.Timestamp(year=self.cur_year, month=1, day=1, hour=12)
        for index, row in self.files[0].iterrows():
            dct[row[HOME_TEAM_COL]] = t
            dct[row[AWAY_TEAM_COL]] = t

        # hack
        if self.cur_year == 2017:
            dct['Buccaneers'] = t
            dct['Dolphins'] = t

        def distance_previous_game(row, prev_week, cur_week, is_home):
            diff = (cur_week[row[HOME_TEAM_COL]] - prev_week[row[HOME_TEAM_COL]]).days if is_home \
                    else (cur_week[row[AWAY_TEAM_COL]] - prev_week[row[AWAY_TEAM_COL]]).days
            if diff == 7:
                return 0
            return 1 if diff > 7 else 2

        for file in self.files:
            dct1 = {}
            for index, row in file.iterrows():
                dct1[row[HOME_TEAM_COL]] = row[GAME_DATE_COL]
                dct1[row[AWAY_TEAM_COL]] = row[GAME_DATE_COL]

            home_prev = file.apply(distance_previous_game, args=(dct, dct1, True), axis=1)
            file[HOME_PREVIOUS_GAME] = home_prev

            away_prev = file.apply(distance_previous_game, args=(dct, dct1, False,), axis=1)
            file[AWAY_PREVIOUS_GAME] = away_prev

            dct.update(dct1)
        return

    def pagerank(self, method, prev_years, prev_weeks, col1, col2, alpha=0.85):

        def create_column_rank(row, pr, col_name):
            if row[col_name] in pr:
                return pr[row[col_name]]
            return None

        def compute_pagerank(H, teams):
            # check for dangling node and update
            num_teams = len(H)
            if np.any(H.sum(axis=1) == 0.0):
                idx = np.argwhere(H.sum(axis=1) == 0.0).flatten()
                for num in idx:
                    H[num] = np.divide(np.ones(num_teams, dtype=float), num_teams)

            # convert H to a row stochastic matrix - S
            S = np.divide(H, H.sum(axis=1)[:, np.newaxis])

            # create column vector of 1's
            y = np.ones(num_teams, dtype=float)[:, np.newaxis]

            # create row vector of 1/n (#nodes)
            v = np.divide(np.ones(num_teams, dtype=float), num_teams)

            # # create the G matrix
            G = ((alpha * S) + ((1 - alpha) * y * v)).T

            # check for convergence
            differences = np.ones(num_teams, dtype=float)[:, np.newaxis]

            v1 = None

            while np.all(differences > 0.00005):
                v1 = np.matmul(G, v)
                differences = np.absolute(v1 - v)
                v = v1

            v1 = np.round(v1, 3)

            return dict(zip(teams, v1))

        # find games played for the previous years and weeks specified
        prev_year_files = [pd.read_csv(self.file_path + str(year) + '/week_' + str(week) + '.csv')
                           for week in range(1, prev_weeks+1)
                           for year in range(self.cur_year-prev_years, self.cur_year+1)]
        games = pd.concat(prev_year_files)

        for file in self.files:
            # construct hmatrix
            hmatrix = hmatrix_factory(method, games, col1, col2)
            H = hmatrix.construct_hmatrix()
            teams = hmatrix.get_teams()

            # compute the pagerank of the previous years
            pr = compute_pagerank(H, teams)

            games = games.append(file)

            home_rank = file.apply(create_column_rank, args=(pr, HOME_TEAM_COL,), axis=1)
            file[col1 + '_' + method + '_rank'] = home_rank

            away_rank = file.apply(create_column_rank, args=(pr, AWAY_TEAM_COL,), axis=1)
            file[col2 + '_' + method + '_rank'] = away_rank
        return

    def create_home_team_win(self):

        def home_team_win(row):
            if row[HOME_SCORE] == row[AWAY_SCORE]:
                return 2
            return 1 if row[HOME_SCORE] > row[AWAY_SCORE] else 0

        for file in self.files:
            home_win = file.apply(home_team_win, axis=1)
            file[HOME_TEAM_WIN] = home_win
        return


if __name__=='__main__':

    years = range(2004, 2020)
    weeks = range(1, 22)
    start_week, end_week = 1, 21
    previous_years = 1

    for year in years:
        files = [pd.read_csv('./data/cleaned_data/' + str(year) + '/week_' + str(week) + '.csv', parse_dates=[GAME_DATE_COL])
                 for week in weeks]

        e = EnrichData(files)

        # create divisional game column
        e.create_divisional_game()

        # create dome game column
        e.create_dome_game()

        # long/short week
        e.create_previous_game_played()

        # create distance away team has travelled
        e.create_away_distance_travelled()

        # margin score
        e.pagerank('margin', previous_years, end_week, HOME_SCORE, AWAY_SCORE)

        # margin passing yards (offence)
        e.pagerank('margin', previous_years, end_week, HOME_PASS_YARDS, AWAY_PASS_YARDS)

        # margin reverse passing yards (defence)
        e.pagerank('margin_reverse', previous_years, end_week, HOME_PASS_YARDS, AWAY_PASS_YARDS)

        # margin rushing yards (offence)
        e.pagerank('margin', previous_years, end_week, HOME_RUSH_YARDS, AWAY_RUSH_YARDS)

        # margin reverse rushing yards (defence)
        e.pagerank('margin_reverse', previous_years, end_week, HOME_RUSH_YARDS, AWAY_RUSH_YARDS)

        # margin turnovers (giveaways)
        e.pagerank('margin', previous_years, end_week, HOME_TURNOVERS, AWAY_TURNOVERS)

        # margin turnovers (takeaways)
        e.pagerank('margin_reverse', previous_years, end_week, HOME_TURNOVERS, AWAY_TURNOVERS)

        # margin sacks (against)
        e.pagerank('margin', previous_years, end_week, HOME_SACKS_ALLOWED, AWAY_SACKS_ALLOWED)

        # margin sacks (for)
        e.pagerank('margin_reverse', previous_years, end_week, HOME_SACKS_ALLOWED, AWAY_SACKS_ALLOWED)

        # margin penalty yards (giveaways)
        e.pagerank('margin', previous_years, end_week, HOME_PENALTY, AWAY_PENALTY)

        # margin penalty yards (given to opponents)
        e.pagerank('margin_reverse', previous_years, end_week, HOME_PENALTY, AWAY_PENALTY)

        # 3rd downs
        e.pagerank('margin', previous_years, end_week, HOME_THIRD_DOWNS_CONVERTED, AWAY_THIRD_DOWNS_CONVERTED)

        # margin reverse 3rd downs
        e.pagerank('margin_reverse', previous_years, end_week, HOME_THIRD_DOWNS_CONVERTED, AWAY_THIRD_DOWNS_CONVERTED)

        # margin 4th downs
        e.pagerank('margin', previous_years, end_week, HOME_FOURTH_DOWNS_CONVERTED, AWAY_FOURTH_DOWNS_CONVERTED)

        # margin reverse 4th downs
        e.pagerank('margin_reverse', previous_years, end_week, HOME_FOURTH_DOWNS_CONVERTED, AWAY_FOURTH_DOWNS_CONVERTED)

        # time of possession (for)
        e.pagerank('margin', previous_years, end_week, HOME_TIME_OF_POSSESSION, AWAY_TIME_OF_POSSESSION)

        # time of possession (against)
        e.pagerank('margin_reverse', previous_years, end_week, HOME_TIME_OF_POSSESSION, AWAY_TIME_OF_POSSESSION)

        # create home team win col
        e.create_home_team_win()

        for i, df in enumerate(e.files, 1):

            fi = df[['game_date', 'year', 'week', 'home_team_alias', 'away_team_alias', 'home_score', 'away_score',
                     'stadium', 'degrees', 'wind', 'humidity', 'roof', 'handicap', 'points_line', 'points_line_result',
                     'is_divisional_game', 'is_dome_game', 'home_previous_game', 'away_previous_game',
                     'away_distance_travelled', 'home_score_margin_rank', 'away_score_margin_rank',
                     'home_pass_yards_margin_rank', 'away_pass_yards_margin_rank', 'home_pass_yards_margin_reverse_rank',
                     'away_pass_yards_margin_reverse_rank', 'home_rush_yards_margin_rank', 'away_rush_yards_margin_rank',
                     'home_rush_yards_margin_reverse_rank', 'away_rush_yards_margin_reverse_rank', 'home_turnovers_margin_rank',
                     'away_turnovers_margin_rank', 'home_turnovers_margin_reverse_rank', 'away_turnovers_margin_reverse_rank',
                     'home_sacks_allowed_margin_rank', 'away_sacks_allowed_margin_rank', 'home_sacks_allowed_margin_reverse_rank',
                     'away_sacks_allowed_margin_reverse_rank', 'home_penallity_yards_margin_rank',
                     'away_penallity_yards_margin_rank', 'home_penallity_yards_margin_reverse_rank',
                     'away_penallity_yards_margin_reverse_rank', 'home_third_downs_converted_margin_rank',
                     'away_third_downs_converted_margin_rank', 'home_third_downs_converted_margin_reverse_rank',
                     'away_third_downs_converted_margin_reverse_rank', 'home_fourth_downs_converted_margin_rank',
                     'away_fourth_downs_converted_margin_rank', 'home_fourth_downs_converted_margin_reverse_rank',
                     'away_fourth_downs_converted_margin_reverse_rank', 'home_time_possession_margin_rank',
                     'away_time_possession_margin_rank', 'home_time_possession_margin_reverse_rank',
                     'away_time_possession_margin_reverse_rank', 'home_team_win']]

            fi.to_csv('./data/enriched_data/' + str(year) + '/week_' + str(i) + '.csv', index=False)

        print(year)










