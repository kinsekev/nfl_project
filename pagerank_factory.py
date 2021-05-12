
import numpy as np
import pandas as pd
from config import HOME_TEAM_COL, AWAY_TEAM_COL

np.set_printoptions(threshold=np.inf)


class MarginHMatrix(object):

    def __init__(self, data_frame, col1, col2):
        self.games = data_frame
        self.col1 = col1
        self.col2 = col2
        self.teams = sorted((self.games[HOME_TEAM_COL].append(self.games[AWAY_TEAM_COL]).drop_duplicates()).tolist())
        self.num_teams = len(self.teams)
        self.H = np.zeros((self.num_teams, self.num_teams), dtype=float)

    def construct_hmatrix(self):
        for index, row in self.games.iterrows():
            # find index position of team
            home_index, away_index = self.teams.index(row[HOME_TEAM_COL]), \
                                     self.teams.index(row[AWAY_TEAM_COL])
            # find home, away values
            home, away = int(row[self.col1]), int(row[self.col2])
            # insert values into the h matrix
            if home < away:
                self.H[home_index][away_index] += away - home
            else:
                self.H[away_index][home_index] += home - away
        return self.H

    def get_teams(self):
        return self.teams


class MarginHReverseMatrix(object):

    def __init__(self, data_frame, col1, col2):
        self.games = data_frame
        self.col1 = col1
        self.col2 = col2
        self.teams = sorted((self.games[HOME_TEAM_COL].append(self.games[AWAY_TEAM_COL]).drop_duplicates()).tolist())
        self.num_teams = len(self.teams)
        self.H = np.zeros((self.num_teams, self.num_teams), dtype=float)

    def construct_hmatrix(self):
        for index, row in self.games.iterrows():
            # find index position of team
            home_index, away_index = self.teams.index(row[HOME_TEAM_COL]), \
                                     self.teams.index(row[AWAY_TEAM_COL])
            # find home, away values
            home, away = int(row[self.col1]), int(row[self.col2])
            # insert values into the h matrix
            if home < away:
                self.H[away_index][home_index] += away - home
            else:
                self.H[home_index][away_index] += home - away
        return self.H

    def get_teams(self):
        return self.teams


class BiDirectionalHMatrix(object):

    def __init__(self, data_frame, col1, col2):
        self.games = data_frame
        self.col1 = col1
        self.col2 = col2
        self.teams = sorted((self.games[HOME_TEAM_COL].append(self.games[AWAY_TEAM_COL]).drop_duplicates()).tolist())
        self.num_teams = len(self.teams)
        self.H = np.zeros((self.num_teams, self.num_teams), dtype=float)

    def construct_hmatrix(self):
        for index, row in self.games.iterrows():
            # find index position of team
            home_index, away_index = self.teams.index(row[HOME_TEAM_COL]), \
                                     self.teams.index(row[AWAY_TEAM_COL])
            # find home, away values
            home, away = int(row[self.col1]), int(row[self.col2])
            # insert values into the h matrix
            self.H[away_index][home_index] += home
            self.H[home_index][away_index] += away
        return self.H

    def get_teams(self):
        return self.teams


class BiDirectionReverseHMatrix(object):

    def __init__(self, data_frame, col1, col2):
        self.games = data_frame
        self.col1_name = col1
        self.col2_name = col2
        self.teams = sorted((self.games[HOME_TEAM_COL].append(self.games[AWAY_TEAM_COL]).drop_duplicates()).tolist())
        self.num_teams = len(self.teams)
        self.H = np.zeros((self.num_teams, self.num_teams), dtype=float)

    def construct_hmatrix(self):
        for index, row in self.games.iterrows():
            # find index position of team
            home_index, away_index = self.teams.index(row[HOME_TEAM_COL]), \
                                     self.teams.index(row[AWAY_TEAM_COL])
            # find home, away values
            home, away = int(row[self.col1_name]), int(row[self.col2_name])
            # insert values into the h matrix
            self.H[home_index][away_index] += home
            self.H[away_index][home_index] += away
        return self.H

    def get_teams(self):
        return self.teams


def hmatrix_factory(method, dataframe, col1, col2):
    if method == 'margin':
        hmatrix = MarginHMatrix
    elif method == 'margin_reverse':
        hmatrix = MarginHReverseMatrix
    elif method == 'bi_directional':
        hmatrix = BiDirectionalHMatrix
    elif method == 'bi_directional_reverse':
        hmatrix = BiDirectionReverseHMatrix
    else:
        raise ValueError(f'Cannot create H matrix with {method}')
    return hmatrix(dataframe, col1, col2)
