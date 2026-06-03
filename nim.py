import math
import random
import time
from typing import List, Tuple, Dict, Optional, Set

class Nim:
    """Class representing a game of Nim."""
    
    def __init__(self, initial: List[int] = [1, 3, 5, 7]):
        """
        Initialize game board.
        Each game board has:
            - piles: a list of how many elements remain in each pile
            - player: 0 or 1, indicating which player to move
            - winner: None if game not over, otherwise 0 or 1
        """
        self.piles: List[int] = initial.copy()
        self.player: int = 0
        self.winner: Optional[int] = None

    @classmethod
    def available_actions(cls, piles: List[int]) -> Set[Tuple[int, int]]:
        """
        Compute available actions in a state.
        action = (pile, count) meaning remove count tokens from pile.
        """
        actions = set()
        for i, pile in enumerate(piles):
            for count in range(1, pile + 1):
                actions.add((i, count))
        return actions

    @staticmethod
    def other_player(player: int) -> int:
        """Return the other player."""
        return 0 if player == 1 else 1

    def switch_player(self) -> None:
        """Switch the current player to the other player."""
        self.player = Nim.other_player(self.player)

    def move(self, action: Tuple[int, int]) -> None:
        """
        Make a move in the game.
        action: (pile, count) means remove count tokens from pile.
        """
        pile, count = action

        # Check for valid move
        if self.winner is not None:
            raise Exception("Game already won")
        if pile < 0 or pile >= len(self.piles):
            raise Exception("Invalid pile")
        if count < 1 or count > self.piles[pile]:
            raise Exception("Invalid number of objects")

        # Update pile
        self.piles[pile] -= count
        self.switch_player()

        # Check for winner
        if all(pile == 0 for pile in self.piles):
            self.winner = self.player


class NimAI:
    """AI for Nim using Q-learning."""
    
    def __init__(self, alpha: float = 0.5, epsilon: float = 0.1):
        """
        Initialize AI with an empty Q-learning dictionary,
        an alpha (learning) rate, and an epsilon rate.
        """
        self.q: Dict[Tuple[Tuple[int, ...], Tuple[int, int]], float] = {}
        self.alpha: float = alpha
        self.epsilon: float = epsilon

    def update(self, old_state: List[int], action: Tuple[int, int], 
               new_state: List[int], reward: int) -> None:
        """
        Update Q-learning model, given an old state, an action taken
        in that state, a new resulting state, and the reward received.
        """
        old_q = self.get_q_value(old_state, action)
        best_future = self.best_future_reward(new_state)
        self.update_q_value(old_state, action, old_q, reward, best_future)

    def get_q_value(self, state: List[int], action: Tuple[int, int]) -> float:
        """
        Return the Q-value for the state `state` and the action `action`.
        If no Q-value exists yet in self.q, return 0.
        """
        # Convert state to tuple so it can be used as dictionary key
        state_tuple = tuple(state)
        key = (state_tuple, action)
        
        # Return Q-value if exists, otherwise 0
        return self.q.get(key, 0.0)

    def update_q_value(self, state: List[int], action: Tuple[int, int], 
                       old_q: float, reward: int, future_rewards: float) -> None:
        """
        Update the Q-value for the state `state` and the action `action`
        given the previous Q-value `old_q`, a current reward `reward`,
        and an estimate of future rewards `future_rewards`.

        Use the formula:
        Q(s, a) <- old value estimate + alpha * (new value estimate - old value estimate)
        where new value estimate = reward + future_rewards
        """
        # Calculate new value estimate
        new_value = reward + future_rewards
        
        # Update Q-value using Q-learning formula
        updated_q = old_q + self.alpha * (new_value - old_q)
        
        # Store updated Q-value
        state_tuple = tuple(state)
        key = (state_tuple, action)
        self.q[key] = updated_q

    def best_future_reward(self, state: List[int]) -> float:
        """
        Given a state `state`, consider all possible `(state, action)`
        pairs available in that state and return the maximum of all
        of their Q-values.

        Use 0 as the Q-value if a particular state-action pair has
        no Q-value in self.q. If there are no available actions in
        the state, return 0.
        """
        # Get all available actions in this state
        actions = Nim.available_actions(state)
        
        if not actions:
            return 0.0
        
        # Find maximum Q-value among all actions
        max_q = float('-inf')
        state_tuple = tuple(state)
        
        for action in actions:
            key = (state_tuple, action)
            q_value = self.q.get(key, 0.0)
            max_q = max(max_q, q_value)
        
        return max_q

    def choose_action(self, state: List[int], epsilon: bool = True) -> Tuple[int, int]:
        """
        Given a state `state`, return an action `(i, j)` to take.

        If `epsilon` is False, then return the best action
        available in the state (the one with the highest Q-value,
        using 0 for pairs that have no Q-values).

        If `epsilon` is True, then with probability `self.epsilon`
        choose a random available action, otherwise choose the best action.

        If multiple actions have the same Q-value, any of those
        options is acceptable.
        """
        # Get all available actions
        actions = list(Nim.available_actions(state))
        
        if not actions:
            raise Exception("No available actions")
        
        state_tuple = tuple(state)
        
        # Epsilon-greedy: with probability epsilon, choose random action
        if epsilon and random.random() < self.epsilon:
            return random.choice(actions)
        
        # Otherwise, choose best action (greedy)
        best_action = None
        best_value = float('-inf')
        
        for action in actions:
            key = (state_tuple, action)
            q_value = self.q.get(key, 0.0)
            
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        # If best_action is still None, choose random
        if best_action is None:
            return random.choice(actions)
        
        return best_action


def train(n: int) -> NimAI:
    """
    Train an AI by playing `n` games against itself.
    """
    player = NimAI()

    # Play n games
    for i in range(n):
        print(f"Playing training game {i + 1}")
        game = Nim()

        # Keep track of last move made by either player
        last = {
            0: {"state": None, "action": None},
            1: {"state": None, "action": None}
        }

        while True:

            # Get available actions
            action = player.choose_action(game.piles)

            # Remember last state and action
            last[game.player]["state"] = game.piles.copy()
            last[game.player]["action"] = action

            # Make move
            game.move(action)
            new_state = game.piles.copy()

            # If game is over, handle rewards
            if game.winner is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    -1
                )
                player.update(
                    last[Nim.other_player(game.player)]["state"],
                    last[Nim.other_player(game.player)]["action"],
                    new_state,
                    1
                )
                break

            # If game continues, no rewards yet
            elif last[game.player]["state"] is not None:
                player.update(
                    last[game.player]["state"],
                    last[game.player]["action"],
                    new_state,
                    0
                )

    print("Done training")
    return player


def play(ai: NimAI, human_player: Optional[int] = None) -> None:
    """
    Play human game against the AI.
    `human_player` can be set to 0 or 1 to specify whether human player moves
    first or second. If None, human player goes first.
    """
    if human_player is None:
        human_player = random.randint(0, 1)

    # Create new game
    game = Nim()

    # Game loop
    while True:
        # Print piles
        print("\nPiles:")
        for i, pile in enumerate(game.piles):
            print(f"Pile {i}: {pile}")

        # Get available actions
        available = Nim.available_actions(game.piles)
        time.sleep(1)

        # Let human make move
        if game.player == human_player:
            print("Your Turn")
            
            # Get valid input from user
            while True:
                try:
                    pile_input = input("Choose pile: ").strip()
                    if not pile_input:
                        print("Please enter a number")
                        continue
                    pile = int(pile_input)
                    
                    count_input = input("Choose count: ").strip()
                    if not count_input:
                        print("Please enter a number")
                        continue
                    count = int(count_input)
                    
                    if (pile, count) in available:
                        break
                    else:
                        print("Invalid move, try again.")
                except ValueError:
                    print("Please enter valid numbers")
                except KeyboardInterrupt:
                    print("\nGame ended")
                    return

        # Let AI make move
        else:
            print("AI's Turn")
            pile, count = ai.choose_action(game.piles, epsilon=False)
            print(f"AI chose to take {count} from pile {pile}.")

        # Make move
        game.move((pile, count))

        # Check for winner
        if game.winner is not None:
            print("\nGAME OVER")
            winner = "Human" if game.winner == human_player else "AI"
            print(f"Winner is {winner}")
            return


if __name__ == "__main__":
    # Train AI
    print("Training AI...")
    ai = train(100)
    
    # Play against AI
    print("\nReady to play!")
    play(ai)
