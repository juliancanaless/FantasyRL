import pandas as pd
from fantasyTeam import Team
import torch
from typing import Optional
import numpy as np

earlyRoundThreshold = 4
middleRoundThreshold = 8
earlyLateRoundThreshold = 12
midLateRoundThreshold = 14

# limits to roster
max_positions = {'QB': 3, 'RB': 5, 'WR': 5, 'TE': 3, 'DST': 2, 'K': 2}
required_positions = {'QB': 1, 'RB': 3, 'WR': 3, 'TE': 2, 'DST': 1, 'K': 1}

stratsByPos = {'QB': ['EarlyRoundQB', 'MidRoundQB', 'LateRoundQB'], 
               'TE': ['EarlyRoundTE', 'MidRoundTE', 'LateRoundTE'], 
               'RB': ['ZeroRB', 'HeroRB'], 
               'WR': ['ZeroWR'],
               'K': ['EarlyK', 'MidK', 'LateK'],
               'DST': ['EarlyDST', 'MidDST', 'LateDST']
               }


# strats by stage
stratsByStage = {'early': ['HeroRB', 'EarlyRoundQB', 'EarlyRoundTE'],
                 'middle': ['MidRoundQB', 'MidRoundTE'],
                 'earlyLate': ['LateRoundQB', 'LateRoundTE', 'EarlyK', 'EarlyDST'],
                 'midLate': ['MidK', 'MidDST'],
                 'lateLate': ['LateK', 'LateDST']
                }


class DraftSimulator:

    def __init__(self, path: str, myTeam: Team, leagueMembers: list, leagueSize: int, numRounds: int, stats: str) -> None:
        self.me = myTeam
        self.teams = []
        for team in leagueMembers:
            entryTeam = Team(team[0], team[1])
            self.teams.append(entryTeam)
        self.teams.append(self.me)
        self.teams.sort(key=lambda team: team.draftPick)
        self.draftBoard = self._prepareBoard(path)
        self.currentRound = 1
        self.currentPick = 1
        self.numRounds = numRounds
        self.numTeams = leagueSize
        self.draftPicksBoard = self._constructTeamPicksBoard()
        self.stats = pd.read_csv(stats)
    
    def _prepareBoard(self, path: str):
        """
        Prepares the draft board of all players available in the draft.
        The ordering is based on the Average Draft Position (ADP) of the players.
        """
        draftBoard = pd.read_csv(path)
        if 'ppr-adp-2024' in path:
            draftBoard['Status'] = 'ACT'
        draftBoard['Available'] = True
        draftBoard = draftBoard.astype(
            {
                'Name': 'string',
                'Team': 'string',
                'ByeWeek': 'Int64',
                'Position': 'string',  
                'PositionRank': 'string',
                'AverageDraftPositionPPR': 'float64',
                'Status': 'string',
                'Available': 'bool'
            }
        )
        return draftBoard
    
    def _constructTeamPicksBoard(self):
        """
        Constructs a board of all fantasy teams within the league in order of their 
        draft picks. This is created to sequentially go through all draft picks.
        The picks follow a "snake" pattern.
        """
        teamPicksBoard = []
        curRound = 1
        curPick = 1
        draftOrdering = [team.name for team in self.teams]
        
        while curRound <= self.numRounds:
            for team in draftOrdering:
                teamPicksBoard.append({
                    'team': team,
                    'draftRound': curRound,
                    'draftPick': curPick,
                    'player': 'N/A',
                    'position': 'N/A',
                    'playerTeam': 'N/A'
                })
                curPick += 1
            draftOrdering.reverse()
            curRound += 1

        
        return pd.DataFrame(teamPicksBoard)
    
    def _determineCurrentStage(self):
        '''
        Determine the current stage of the draft.
        The 3 stages are early, middle, and late.
        Early is classified as the first 4 rounds.
        Middle is classified as rounds 5-8.
        Late is classified as rounds 9 and onward.
        '''
        if self.currentRound <= earlyRoundThreshold:
            return 'early'
        elif self.currentRound <= middleRoundThreshold:
            return 'middle'
        elif self.currentRound <= earlyLateRoundThreshold:
            return 'earlyLate'
        elif self.currentRound <= midLateRoundThreshold:
            return 'midLate'
        else:
            return 'lateLate'
    
    def _selectTopPlayerByPositionSet(self, position_set: set, team: Team, stage: str):
        '''
        Given a set that contains positions a specific team is in search for,
        this function returns the highest valued player amongst the positions.
        It also updates internal information for team draft records/strategy.
        The team will draft the top available option. 
        '''
        available_players = self.draftBoard[self.draftBoard['Available'] == True]
        for _, row in available_players.iterrows():
            position = row['Position']
            playerTeam = row['Team']
            byeWeek = row['ByeWeek']
            status = row['Status']
            avgadp = row['AverageDraftPositionPPR']
            if ((position_set and (position in position_set)) or not position_set) \
            and team.posFreqMap[position] < max_positions.get(position):
                for strat in team.strategy:
                    if strat in stratsByStage[stage] and strat in stratsByPos[position]:
                        # we can reduce position req
                        team.picksNeeded[stage] -= 1
                player_name = row['Name']
                self.draftBoard.loc[self.draftBoard['Name'] == player_name, 'Available'] = False
                self.draftPicksBoard.loc[self.draftPicksBoard['draftPick'] == self.currentPick, ['player', 'position', 'playerTeam']] = player_name, position, playerTeam
                return player_name, playerTeam, position, byeWeek, status, avgadp
        return None
    
    def _determineRequiredPositions(self, team: Team, remaining_rounds: int):
        '''
        This function is used in late rounds to determine if there are more than
        enough rounds in the draft to satisfy roster requirements:
        required_positions = {'QB': 1, 'RB': 3, 'WR': 3, 'TE': 2, 'DST': 1, 'K': 1}.
        If there are barely enough, it returns a set of the required positions, otherwise
        returns an empty set.
        '''
        required_positions_set = set()
        num_picks_needed = 0
        for position, required_count in required_positions.items():
            current_count = team.posFreqMap[position]
            if current_count < required_count:
                required_positions_set.add(position)
                num_picks_needed += (required_count - current_count)
        if num_picks_needed >= remaining_rounds:
            return required_positions_set
        return set()
    
    def otherTeamSelection(self, team: Team):
        '''
        This function simulates other teams (not the model we are training) making
        picks based on their strategy and available players. 
        '''
        qb_strat, rb_strat, wr_strat, te_strat, k_strat, dst_strat = team.strategy
        stage = self._determineCurrentStage()
        
        if stage == 'early':
            # determine # of early round strats
            numPicksReserved = team.picksNeeded[stage]
            roundsLeftInStage = earlyRoundThreshold - self.currentRound + 1
            potentialPositions = set()
            if numPicksReserved == roundsLeftInStage:
                # must use all rounds left in stage to satisfy reqs
                if rb_strat == 'HeroRB' and self.currentRound == 1:
                    potentialPositions.add('RB')
                    return self._selectTopPlayerByPositionSet(potentialPositions, team, stage)
                if qb_strat == 'EarlyRoundQB' and team.posFreqMap['QB'] == 0:
                    potentialPositions.add('QB')
                if te_strat == 'EarlyRoundTE' and team.posFreqMap['TE'] == 0:
                    potentialPositions.add('TE')
                return self._selectTopPlayerByPositionSet(potentialPositions, team, stage)
            else:
                # can draft wtv favorable and kick the requirement down the road
                if rb_strat == 'HeroRB' and self.currentRound == 1:
                    potentialPositions.add('RB')
                    return self._selectTopPlayerByPositionSet(potentialPositions, team, stage)
                if qb_strat == 'EarlyRoundQB' and team.posFreqMap['QB'] == 0:
                    potentialPositions.add('QB')
                if te_strat == 'EarlyRoundTE' and team.posFreqMap['TE'] == 0:
                    potentialPositions.add('TE')
                potentialPositions.add('WR')
                potentialPositions.add('RB')
                if rb_strat == 'ZeroRB' or rb_strat == 'HeroRB':
                    potentialPositions.remove('RB')
                if wr_strat == 'ZeroWR':
                    potentialPositions.remove('WR')
                return self._selectTopPlayerByPositionSet(potentialPositions, team, stage)

        elif stage == 'middle':
            # determine # of middle round strats
            numPicksReserved = team.picksNeeded[stage]
            roundsLeftInStage = middleRoundThreshold - self.currentRound + 1
            potentialPositions = set()
            if numPicksReserved == roundsLeftInStage:
                # must use all rounds left in stage to satisfy reqs
                if qb_strat == 'MidRoundQB' and team.posFreqMap['QB'] == 0:
                    potentialPositions.add('QB')
                if te_strat == 'MidRoundTE' and team.posFreqMap['TE'] == 0:
                    potentialPositions.add('TE')
                return self._selectTopPlayerByPositionSet(potentialPositions, team, stage)
            else:
                # can draft wtv favorable and kick the requirement down the road
                if qb_strat == 'MidRoundQB' and team.posFreqMap['QB'] == 0:
                    potentialPositions.add('QB')
                if te_strat == 'MidRoundTE' and team.posFreqMap['TE'] == 0:
                    potentialPositions.add('TE')
                potentialPositions.add('WR')
                potentialPositions.add('RB')
                if te_strat == 'EarlyRoundTE':
                    # can add te to draft set
                    potentialPositions.add('TE')
                # wont draft qb in middle rounds if drafted premium qb in early rounds
                return self._selectTopPlayerByPositionSet(potentialPositions, team, stage)
        elif stage == 'earlyLate':
            # determine # of earlyLate round strats
            numPicksReserved = team.picksNeeded[stage]
            roundsLeftInStage = earlyLateRoundThreshold - self.currentRound + 1
            potentialPositions = set()
            roundsUntilEnd = self.numRounds - self.currentRound + 1
            posReqSet = self._determineRequiredPositions(team, roundsUntilEnd)
            if len(posReqSet) != 0:
                # must adhere all remaining picks to fulfill roster reqs
                return self._selectTopPlayerByPositionSet(posReqSet, team, stage)
            if numPicksReserved == roundsLeftInStage:
                # must use all rounds left in stage to satisfy reqs
                if qb_strat == 'LateQB' and team.posFreqMap['QB'] == 0:
                    potentialPositions.add('QB')
                if te_strat == 'LateTE' and team.posFreqMap['TE'] == 0:
                    potentialPositions.add('TE')
                if k_strat == 'EarlyK' and team.posFreqMap['K'] == 0:
                    potentialPositions.add('K')
                if dst_strat == 'EarlyDST' and team.posFreqMap['DST'] == 0:
                    potentialPositions.add('DST')
                return self._selectTopPlayerByPositionSet(potentialPositions, team, stage)
            else:
                # can draft wtv favorable and kick the requirement down the road
                if qb_strat == 'LateQB' and team.posFreqMap['QB'] == 0:
                    potentialPositions.add('QB')
                if te_strat == 'LateTE' and team.posFreqMap['TE'] == 0:
                    potentialPositions.add('TE')
                if k_strat == 'EarlyK' and team.posFreqMap['K'] == 0:
                    potentialPositions.add('K')
                if dst_strat == 'EarlyDST' and team.posFreqMap['DST'] == 0:
                    potentialPositions.add('DST')
                potentialPositions.add('WR')
                potentialPositions.add('RB')
                if te_strat == 'EarlyRoundTE' or te_strat == 'MidRoundTE':
                    # can add te to draft set
                    potentialPositions.add('TE')
                # wont draft qb in middle rounds if drafted premium qb in early rounds
                return self._selectTopPlayerByPositionSet(potentialPositions, team, stage)
        elif stage == 'midLate':
            # determine # of midLate round strats
            numPicksReserved = team.picksNeeded[stage]
            roundsLeftInStage = midLateRoundThreshold - self.currentRound + 1
            potentialPositions = set()
            roundsUntilEnd = self.numRounds - self.currentRound + 1
            posReqSet = self._determineRequiredPositions(team, roundsUntilEnd)
            if len(posReqSet) != 0:
                # must adhere all remaining picks to fulfill roster reqs
                return self._selectTopPlayerByPositionSet(posReqSet, team, stage)
            if numPicksReserved == roundsLeftInStage:
                # must use all rounds left in stage to satisfy reqs
                if k_strat == 'MidK' and team.posFreqMap['K'] == 0:
                    potentialPositions.add('K')
                if dst_strat == 'MidDST' and team.posFreqMap['DST'] == 0:
                    potentialPositions.add('DST')
                return self._selectTopPlayerByPositionSet(potentialPositions, team, stage)
            else:
                # can draft wtv favorable and kick the requirement down the road
                if k_strat == 'MidK' and team.posFreqMap['K'] == 0:
                    potentialPositions.add('K')
                if dst_strat == 'MidDST' and team.posFreqMap['DST'] == 0:
                    potentialPositions.add('DST')
                potentialPositions.add('WR')
                potentialPositions.add('RB')
                potentialPositions.add('TE')
                potentialPositions.add('QB')
                # wont draft qb in middle rounds if drafted premium qb in early rounds
                return self._selectTopPlayerByPositionSet(potentialPositions, team, stage)
        else:
            # determine # of late round strats
            numPicksReserved = team.picksNeeded[stage] 
            roundsLeftInStage = self.numRounds - self.currentRound + 1
            potentialPositions = set()
            posReqSet = self._determineRequiredPositions(team, roundsLeftInStage)
            if len(posReqSet) != 0:
                # must adhere all remaining picks to fulfill roster reqs
                return self._selectTopPlayerByPositionSet(posReqSet, team, stage)
            else:
                # can draft wtv favorable and kick the requirements (if any) down the road 
                potentialPositions.add('WR')
                potentialPositions.add('RB')
                potentialPositions.add('QB')
                potentialPositions.add('TE')
                return self._selectTopPlayerByPositionSet(potentialPositions, team, stage)

    def mySelection(self, player_name):
        """
        Simulate me making a selection.
        I will take players based on my deep learning model analysis.
        The model that will be used will be reinforcement training.
        """
        row = self.draftBoard.loc[self.draftBoard['Name'] == player_name]
        position = row['Position'].iloc[0]
        playerTeam = row['Team'].iloc[0]
        byeWeek = row['ByeWeek'].iloc[0]
        status = row['Status'].iloc[0]
        avgadp = row['AverageDraftPositionPPR'].iloc[0]
        self.me.addPickToRoster(position, player_name, self.currentPick, avgadp, playerTeam, byeWeek, 0, status)
        
        self.draftBoard.loc[self.draftBoard['Name'] == player_name, 'Available'] = False

        print(f"My selection at pick {self.currentPick}, round {self.currentRound}: {player_name}, {position}")
        self.currentPick += 1

    def constructWaiverWire(self):
        '''
        Constructs waiver wire out of left over players that went undrafted.
        '''
        waiverWire = self.draftBoard[self.draftBoard['Available'] == True]
        waiverWire.drop(columns=['PositionRank', 'Available'], inplace=True)
        waiverWire['PointsPerGame'] = 0.0
        waiverWire['ProjectedFantasyPoints'] = 0.0
        waiverWire['FantasyPoints'] = 0.0
        return waiverWire

    def get_state(self):
        '''
        Returns the state of the draft (i.e. the draft board)
        '''
        draftboard_state = self.draftBoard.to_numpy(dtype=np.float32)
        return draftboard_state