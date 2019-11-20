"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

##def pathProcess(path, start, goal):
##    """
##    start, goal is (x,y) pair, location of start, goal state
##    path is a dict{location, direction} containing all the states traversed,
##    return the goal path
##    """
##
##    if(len(path) == 0):
##        return []
##    print('a',goal)
##    cur_face = path[goal]
##    cur_x = goal[0]
##    cur_y = goal[1]
##    solution1 = []
##
##    while(1):
##        print((cur_x,cur_y), start)
##        if((cur_x,cur_y) == start):
##            break
##        cur_face = path[(cur_x,cur_y)]
##        solution1.append(cur_face)   
##        if(cur_face == "West"):
##            cur_x += 1
##        if(cur_face == "East"):
##            cur_x -= 1
##        if(cur_face == "South"):
##            cur_y += 1
##        if(cur_face == "North"):
##            cur_y -= 1            
##        
##    solution1.reverse()
##
##    print(solution1)
##    return solution1

def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    closed = set()
    solution = []
    fringe = util.Stack()
    init_path = []
    fringe.push((problem.getStartState(),init_path))
    while(1):
        if(fringe.isEmpty()):
            util.raiseNotDefined()
        (cur_state, cur_path) = fringe.pop()
        if(problem.isGoalState(cur_state)):
            return cur_path
        if(cur_state not in closed):
            closed.add(cur_state)
            for each in problem.getSuccessors(cur_state):
                if(not cur_path):
                    cur_path = []
                next_path = cur_path + [each[1]]
                fringe.push((each[0],next_path))
    util.raiseNotDefined()
    
def depthFirstSearch1(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    closed = []
##if use set, state ((x,y),[False,False,False,False]) becomes unhashable in 
    solution = []
    fringe = util.Stack()
    fringe.push(problem.getStartState())
    path = dict()
    while(1):
        if(fringe.isEmpty()):
            return [] 
        state = fringe.pop()
        if(problem.isGoalState(state)):
            return pathProcess(path, problem.getStartState(), state)
        if(state not in closed):
            closed.add(state)
            for each in problem.getSuccessors(state):
                if(each[0] not in closed):
                    path[each[0]] = each[1]
                fringe.push(each[0])
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    closed = set()
    solution = []
    fringe = util.Queue()
    init_path = []
    fringe.push((problem.getStartState(),init_path))
    while(1):
        if(fringe.isEmpty()):
            util.raiseNotDefined()
        (cur_state, cur_path) = fringe.pop()
        if(problem.isGoalState(cur_state)):
            return cur_path
        if(cur_state not in closed):
            closed.add(cur_state)
            for each in problem.getSuccessors(cur_state):
                if(not cur_path):
                    cur_path = []
                next_path = cur_path + [each[1]]
                fringe.push((each[0],next_path))
    util.raiseNotDefined()

def costSmaller(s1, s2):
    if s1[2] < s2[2]:
        return True
    else:
        return False
    
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    closed = set()
    solution = []
    fringe = util.PriorityQueue()
    init_path = []
    init_cost = 0
    init_priority = 0
    fringe.update((problem.getStartState(),init_path, init_cost), init_priority)
## Here cost is the priority
    while(1):
        if(fringe.isEmpty()):
            util.raiseNotDefined()
        (cur_state,cur_path,cur_cost) = fringe.pop()
        
        if(problem.isGoalState(cur_state)):
            return cur_path
        if(cur_state not in closed):
            closed.add(cur_state)
            for each in problem.getSuccessors(cur_state):
                if(not cur_path):
                    cur_path = []
                next_path = cur_path + [each[1]]
                next_cost = cur_cost + each[2]
                fringe.update((each[0],next_path,next_cost),next_cost)
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    closed = set()
    solution = []
    fringe = util.PriorityQueue()
    init_path = []
    init_cost = 0
    init_priority = heuristic(problem.getStartState(),problem)
    fringe.update((problem.getStartState(),init_path, init_cost), init_priority)
    while(1):
        if(fringe.isEmpty()):
            util.raiseNotDefined()
        (cur_state,cur_path,cur_cost) = fringe.pop()
        
        if(problem.isGoalState(cur_state)):
            return cur_path
        if(cur_state not in closed):
            closed.add(cur_state)
            for each in problem.getSuccessors(cur_state):
                if(not cur_path):
                    cur_path = []
                next_path = cur_path + [each[1]]
                next_cost = cur_cost + each[2]
                next_priority = cur_cost + each[2] + heuristic(each[0], problem)
####
##                if next_priority < heuristic(cur_state, problem):
##                    print("Inconsistent!")
##                    util.raiseNotDefined()
####

                fringe.update((each[0],next_path,next_cost),next_priority)
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
