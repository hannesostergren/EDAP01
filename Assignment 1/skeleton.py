import gym
import random
import requests
import numpy as np
import argparse
import copy
import time
import sys
from gym_connect_four import ConnectFourEnv

env: ConnectFourEnv = gym.make("ConnectFour-v0")

#SERVER_ADRESS = "http://localhost:8000/"
SERVER_ADRESS = "https://vilde.cs.lth.se/edap01-4inarow/"
API_KEY = 'nyckel'
STIL_ID = ["ma3875he-s","ha6882os-s"] # TODO: fill this list with your stil-id's

def call_server(move):
   res = requests.post(SERVER_ADRESS + "move",
                       data={
                           "stil_id": STIL_ID,
                           "move": move, # -1 signals the system to start a new game. any running game is counted as a loss
                           "api_key": API_KEY,
                       })
   # For safety some respose checking is done here
   if res.status_code != 200:
      print("Server gave a bad response, error code={}".format(res.status_code))
      exit()
   if not res.json()['status']:
      print("Server returned a bad status. Return message: ")
      print(res.json()['msg'])
      exit()
   return res

def check_stats():
   res = requests.post(SERVER_ADRESS + "stats",
                       data={
                           "stil_id": STIL_ID,
                           "api_key": API_KEY,
                       })

   stats = res.json()
   return stats

"""
You can make your code work against this simple random agent
before playing against the server.
It returns a move 0-6 or -1 if it could not make a move.
To check your code for better performance, change this code to
use your own algorithm for selecting actions too
"""
def opponents_move(env):
   env.change_player() # change to oppoent
   avmoves = env.available_moves()
   if not avmoves:
      env.change_player() # change back to student before returning
      return -1

   # TODO: Optional? change this to select actions with your policy too
   # that way you get way more interesting games, and you can see if starting
   # is enough to guarrantee a win
   action = random.choice(list(avmoves))

   state, reward, done, _ = env.step(action)
   if done:
      if reward == 1: # reward is always in current players view
         reward = -1
   env.change_player() # change back to student before returning
   return state, reward, done


def zugzwang(env):
   print("hello")

def max_function(env, alpha, beta, depth,t1): 
   if depth == 0 or env.is_win_state() or time.time()-t1>4.7:
      return eval(env)
   v = - np.inf
   for successorMove in list(env.available_moves()):
      successorState = copy.deepcopy(env)
      successorState.step(successorMove)
      successorState.change_player()

      v = max(v,min_function(successorState, alpha, beta, depth-1,t1))
      alpha = max(alpha,v)
      
      if v >= beta: 
         return v 
      
   return v


def min_function(env, alpha, beta, depth,t1): 
   if depth == 0 or env.is_win_state()or time.time()-t1>4.7:
      return eval(env)
   v = np.inf
   for successorMove in list(env.available_moves()):

      successorState = copy.deepcopy(env)
      successorState.step(successorMove)
      successorState.change_player()
      v =min(v,max_function(successorState, alpha, beta, depth-1,t1))
      beta = min(beta,v)
      if v <= alpha: 
         return v 
      
   return v

def longestLine(env):
        # Test rows
   
   p1featureValue = 0 
   center_array = [int(i) for i in list(env.board[:,env.board_shape[0]//2])]
   center_count = center_array.count(1)
   p1featureValue += center_count*3
   #print(env.board_shape)

   for i in range(env.board_shape[0]):
      for j in range(env.board_shape[1] - 3):
         prevSquares = []
         for y in range(4):
            square = env.board[i][j+y]
            prevSquares.append(square) 
         p1featureValue += eval4(prevSquares,1)
        # Test columns on transpose array

   reversed_board = [list(i) for i in zip(*env.board)]
   for i in range(env.board_shape[1]):
      for j in range(env.board_shape[0] - 3):
         prevSquares = []
         for y in range(4):
            square =reversed_board[i][j+y]
            prevSquares.append(square) 
         p1featureValue += eval4(prevSquares,1)
      
        # Test diagonal

   for i in range(env.board_shape[0] - 3):
      for j in range(env.board_shape[1] - 3):
         prevSquares = []
         for k in range(4):
            square = env.board[i + k][j + k]
            prevSquares.append(square) 
         p1featureValue += eval4(prevSquares,1)
      


   reversed_board = np.fliplr(env.board)
        # Test reverse diagonal
   for i in range(env.board_shape[0] - 3):
      for j in range(env.board_shape[1] - 3):
         prevSquares = []
         for k in range(4):
            square = reversed_board[i + k][j + k]
            prevSquares.append(square) 
         p1featureValue += eval4(prevSquares,1)



   return p1featureValue

def eval4(squares, player:int):
   if sum(squares) == 4:
      return 1000
   elif sum(squares) == 3 and -1 not in squares:
      return 5
   elif sum(squares) == 2 and -1 not in squares:
      return 2
   elif sum(squares) == -3 and 1 not in squares:
      return -4
   elif sum(squares) == -4:
      return -1000
   else:
      return 0

def eval(env): 
   p1featureValue = longestLine(env)
   return p1featureValue 

def student_move(env,moveNum):
   if moveNum == 1:
      return env.board_shape[0]//2
   initialAlpha = - np.inf
   initialBeta = np.inf
   maxDepth = 4
   bestAction = None
   avMovs = list(env.available_moves())
   maxScores = []
   t1 = time.time()
   for move in avMovs:      
      t2 = time.time()
      if t2-t1 > 4.8:
         break
      successorState = copy.deepcopy(env)
      successorState.step(move)
      successorState.change_player()
      score = min_function(successorState, initialAlpha,initialBeta,maxDepth,t1)
      if move == avMovs[0]:
         maxScore = score
      
      #print(score)
      if score == maxScore:
         maxScores.append(move)
      if score > maxScore:
         maxScores = []   
         maxScore=score
         bestAction = move
   if len(maxScores) > 0:
      bestAction = random.choice(maxScores)
   return bestAction

def play_game(vs_server = False):
   """
   The reward for a game is as follows. You get a
   botaction = random.choice(list(avmoves)) reward from the
   server after each move, but it is 0 while the game is running
   loss = -1
   win = +1
   draw = +0.5
   error = -10 (you get this if you try to play in a full column)
   Currently the player always makes the first move
   """

   # default state
   state = np.zeros((6, 7), dtype=int)

   # setup new game
   if vs_server:
      # Start a new game
      res = call_server(-1) # -1 signals the system to start a new game. any running game is counted as a loss

      # This should tell you if you or the bot starts
      print(res.json()['msg'])
      botmove = res.json()['botmove']
      state = np.array(res.json()['state'])
      env.reset(board=state)
   else:
      # reset game to starting state
      env.reset(board=None)
      # determine first player
      student_gets_move = random.choice([True, False])
      if student_gets_move:
         print('You start!')
         print()
      else:
         print('Bot starts!')
         print()

   # Print current gamestate
   print("Current state (1 are student discs, -1 are servers, 0 is empty): ")
   print(state,env)
   #print("After Opening ----------------------------")

   done = False
   moves = 0

   while not done:
      # Select your move
      moves +=1
      t1 = time.time()
      stmove = student_move(env,moves) 
      t2 = time.time()

      # make both student and bot/server moves
      if vs_server:
         # Send your move to server and get response
         env.step(stmove)
         res = call_server(stmove)
         print(res.json()['msg'])

         # Extract response values
         result = res.json()['result']
         botmove = res.json()['botmove']
         state = np.array(res.json()['state'])
         env.change_player()
         env.step(botmove)
         env.change_player()

      else:
         if student_gets_move:
            # Execute your move
            avmoves = env.available_moves()
            if stmove not in avmoves:
               print("You tried to make an illegal move! You have lost the game.")
               break
            state, result, done, _ = env.step(stmove)

         student_gets_move = True # student only skips move first turn if bot starts

         # print or render state here if you like

         # select and make a move for the opponent, returned reward from students view
         if not done:
            state, result, done = opponents_move(env)

      # Check if the game is over
      if result != 0:
         done = True
         print(state)
         if not vs_server:
            print("Game over. ", end="")
         if result == 1:
            print("You won!")
         elif result == 0.5:
            print("It's a draw!")
         elif result == -1:
            print("You lost!")
         elif result == -10:
            print("You made an illegal move and have lost!")
         else:
            print("Unexpected result result={}".format(result))
         if not vs_server:
            print("Final state (1 are student discs, -1 are servers, 0 is empty): ")
      else:

         print("Current state (1 are student discs, -1 are servers, 0 is empty): ")

      # Print current gamestate
      print(state)
      print("Time of move",  t2-t1)
      print()

def main():
   # Parse command line arguments
   parser = argparse.ArgumentParser()
   group = parser.add_mutually_exclusive_group()
   group.add_argument("-l", "--local", help = "Play locally", action="store_true")
   group.add_argument("-o", "--online", help = "Play online vs server", action="store_true")
   parser.add_argument("-s", "--stats", help = "Show your current online stats", action="store_true")
   args = parser.parse_args()

   # Print usage info if no arguments are given
   if len(sys.argv)==1:
      parser.print_help(sys.stderr)
      sys.exit(1)

   if args.local:
      play_game(vs_server = False)
   elif args.online:
      play_game(vs_server = True)

   if args.stats:
      stats = check_stats()
      print(stats)

   # TODO: Run program with "--online" when you are ready to play against the server
   # the results of your games there will be logged
   # you can check your stats bu running the program with "--stats"

if __name__ == "__main__":
    main()