import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        states = mdp.getStates()
        for it_num in range(self.iterations):
            new_val = []
            for i in range(len(states)):
                state = states[i]
                actions = mdp.getPossibleActions(state)
                max_qval = -float("inf")
                for j in range(len(actions)):
                    action = actions[j]
                    next_state_pairs = mdp.getTransitionStatesAndProbs(state, action)
                    qval = self.computeQValueFromValues(state,action)
                    if qval>max_qval:
                        max_qval = qval
                new_val.append(max_qval)
            for i in range(len(states)):
                self.values[states[i]] = new_val[i]
                if states[i] == 'TERMINAL_STATE':
                    self.values[states[i]] = 0
                

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        next_state_pairs = mdp.getTransitionStatesAndProbs(state, action)
        qval = 0
        for i in range(len(next_state_pairs)):
            next_state = next_state_pairs[i][0]
            reward = mdp.getReward(state, action, next_state)
            prob = next_state_pairs[i][1]
            if not len(mdp.getPossibleActions(state)):
            	qval += prob*reward
            	continue 
            next_val = self.getValue(next_state)
            qval += prob * (reward + self.discount*next_val)
        return qval;

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        actions = mdp.getPossibleActions(state)
        if not len(actions):
            return None
        max_qval = -float("inf")
        location = 0
        for i in range(len(actions)):
            qval = self.computeQValueFromValues(state, actions[i])
            if qval > max_qval:
                max_qval = qval
                location = i
        return actions[location];  

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        states = mdp.getStates()
        states_num = len(states)
        state_index = 0
        
        for it_num in range(self.iterations):
            state = states[state_index]
            if state == 'TERMINAL_STATE':
                state_index = (state_index + 1) % states_num
                continue

            actions = mdp.getPossibleActions(state)
            max_qval = -float("inf")
            for j in range(len(actions)):
                action = actions[j]
                next_state_pairs = mdp.getTransitionStatesAndProbs(state, action)
                qval = self.computeQValueFromValues(state,action)
                if qval>max_qval:
                    max_qval = qval
            self.values[state] = max_qval        
            state_index = (state_index + 1) % states_num

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp = self.mdp
        states = mdp.getStates()
        predecessors = [set() for i in states]
        
        for state in states:
            for action in mdp.getPossibleActions(state):
                next_state_list = [pair[0] for pair in \
                                mdp.getTransitionStatesAndProbs(state, action)]
                for ns in next_state_list:
##                    print(states.index(ns))
                    predecessors[states.index(ns)].add(state)
##        for s in states:
##            print(s,predecessors[states.index(s)])
        pq = util.PriorityQueue()
        diff_list = [0 for i in states]
        for state in states:
            if state == 'TERMINAL_STATE':
                continue
            qvals = [self.computeQValueFromValues(state, action)\
                     for action in mdp.getPossibleActions(state)]
            diff = abs(self.values[state] - max(qvals))
            pq.update(state, -diff)
            diff_list[states.index(state)] = diff
        for it_num in range(self.iterations):
            if pq.isEmpty():
                return
            state = pq.pop()
            if state != 'TERMINAL_STATE':
                qvals = [self.computeQValueFromValues(state, action)\
                     for action in mdp.getPossibleActions(state)]
                self.values[state] = max(qvals)
##                print(self.values[states.index(state)])
            for p in predecessors[states.index(state)]:
                qvals = [self.computeQValueFromValues(p, action)\
                     for action in mdp.getPossibleActions(p)]
                diff = abs(self.values[p] - max(qvals))
##                diff_list[states.index(p)] = diff
                if diff > self.theta:               
                    pq.update(p, -diff)
                    
