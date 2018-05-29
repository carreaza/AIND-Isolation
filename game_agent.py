"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def center_distance(game,player):
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    return float((h - y) ** 2 + (w - x) ** 2)**(1/2)

def player_distance(game,player):
    y_opp, x_opp = game.get_player_location(game.get_opponent(player))
    y, x = game.get_player_location(player)
    return float((y-y_opp) ** 2 + (x-x_opp) ** 2)**(1/2)

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    #my_moves
    num_moves = float(len(game.get_legal_moves(player)))
    num_moves_opp = float(len(game.get_legal_moves(game.get_opponent(player))))
    return float((num_moves - player_distance(game,player) - num_moves_opp))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    num_moves = float(len(game.get_legal_moves(player)))
    num_moves_opp = float(len(game.get_legal_moves(game.get_opponent(player))))
    return (num_moves - center_distance(game,player) - num_moves_opp)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    eval_func = 0.0

    num_moves = float(len(game.get_legal_moves(player)))
    num_moves_opp = float(len(game.get_legal_moves(game.get_opponent(player))))
    
#    if(game.move_count < 10):
#        eval_func += (num_moves - num_moves_opp)
#    
#    else:
#        eval_func += (num_moves - num_moves_opp)
#        eval_func -= center_distance(game,player)
    
    eval_func += (num_moves - center_distance(game,player) - num_moves_opp)        
    #penalizing for being in the corners
    corners = [[0,0],[0,1],[1,0],
               [game.width,game.height],[game.width-1,game.height],[game.width,game.height-1],
               [game.width,0],[game.width-1,0],[game.width,1],
               [0,game.height],[0,game.height-1],[1,game.height]]
    if(game.get_player_location(player) in corners):
        eval_func -= 4
        
    return float(eval_func)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************
[game.width,game.height],
    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score = float("-inf")
        best_move = None
        for m in game.get_legal_moves():
            
            # call has been updated with a depth limit
            v = self.min_value(game.forecast_move(m), depth - 1)
            if v > best_score:
                best_score = v
                best_move = m
        return best_move

    
    def min_value(self, game, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
#        if self.time_left() < self.TIMER_THRESHOLD:
#            raise SearchTimeout()
            
        if self.terminal_test(game):
            return self.score(game,self)  # by Assumption 2
        
        # New conditional depth limit cutoff
        if depth <= 0:  # "==" could be used, but "<=" is safer 
            return self.score(game,self)
        
        v = float("inf")
        for m in game.get_legal_moves():
            # the depth should be decremented by 1 on each call
            v = min(v, self.max_value(game.forecast_move(m), depth - 1))
        return v        
    
    def max_value(self, game, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        if self.terminal_test(game):
            return self.score(game,self)  # by assumption 2
        
        # New conditional depth limit cutoff
        if depth <= 0:  # "==" could be used, but "<=" is safer 
            return self.score(game,self) #pass board and player (as self)
        
        v = float("-inf")
        for m in game.get_legal_moves():
            # the depth should be decremented by 1 on each call
            v = max(v, self.min_value(game.forecast_move(m), depth - 1))
        return v
    
    def terminal_test(self, game):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()        
        # NOTE: do NOT modify this function
#        global call_counter #commented out by Carlos on Mar 27: may not be needed
#        call_counter += 1
        moves_available = bool(game.get_legal_moves())  # by Assumption 1
        return not moves_available

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move = (-1, -1)
        depth_count = 0
        
        while(True):
            try:
                # The try/except block will automatically catch the exception
                # raised when the timer is about to expire.
                best_move = self.alphabeta(game, depth_count)
                depth_count += 1
    
            except SearchTimeout:
                break

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        best_score = float("-inf")
        best_move = [-1,-1]
        v, best_move = self.max_value_ab(game, alpha, beta, depth)
        
        return best_move

    
    def min_value_ab(self, game, alpha, beta, depth):
        """ Return the value for a win (+1) if the game is over,
        otherwise return the minimum value over all legal child
        nodes.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        best_move = [-1,-1]
        
        if self.terminal_test_ab(game):
            return self.score(game,self),best_move  # by Assumption 2
        
        # New conditional depth limit cutoff
        if depth <= 0:  # "==" could be used, but "<=" is safer 
            return self.score(game,self),best_move
        
        v = float("inf")
        
        for m in game.get_legal_moves():

            value, move =  self.max_value_ab(game.forecast_move(m), alpha, beta, depth-1)
            #check for best move
            if(value < v):
                v = value
                best_move = m
            if (v <= alpha):
                return v, best_move
            beta = min(beta,v)
        return v, best_move        
    
    def max_value_ab(self, game, alpha, beta, depth):
        """ Return the value for a loss (-1) if the game is over,
        otherwise return the maximum value over all legal child
        nodes.
        """    
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
            
        best_move = [-1,-1]
        
        if self.terminal_test_ab(game):
            return self.score(game,self), best_move  # by assumption 2
        
        # New conditional depth limit cutoff
        if depth <= 0:  # "==" could be used, but "<=" is safer 
            return self.score(game,self), best_move #pass board and player (as self)
        
        v = float("-inf")
        
        for m in game.get_legal_moves():
            # the depth should be decremented by 1 on each call
            value, move = self.min_value_ab(game.forecast_move(m),alpha,beta,depth-1)
            #check for best move
            if(value > v):
                v = value
                best_move = m
            if (v >= beta):
                return v, best_move
            alpha = max(alpha, v)
        return v, best_move
    
    def terminal_test_ab(self, game):
        """ Return True if the game is over for the active player
        and False otherwise.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()        
        # NOTE: do NOT modify this function
#        global call_counter #commented out by Carlos on Mar 27: may not be needed
#        call_counter += 1
        moves_available = bool(game.get_legal_moves())  # by Assumption 1
        return not moves_available
