from helper_classes import Pair, Ocean
import numpy as np
import functools
import scipy
from exit import Exit


class Agent:
    """
    Agent class: personality traits, emotion preference, emotion contagion
    """

    def __init__(self, id: int):
        self.id = id
        self.source: Pair = Pair(None, None)
        self.destination: Exit = None

        # motion features
        self.velocity: Pair = Pair(None, None)
        self.position: Pair = Pair(None, None)

        self.traits: Ocean = Ocean.empty()
        self.panic_factor = 0
        self.init_distance_preference = self.calculate_init_distance_preference()
        self.init_velocity_preference = self.calculate_init_velocity_preference()

        self.distance_preference = self.init_distance_preference
        self.velocity_preference = self.init_velocity_preference

        self.arrivied = False
        self.colided = False
        self.history: list[Pair] = []
        self.current_panic = 0.0

    def calculate_panic_factor(self):
        # Openness: Cleverness, creativity +- panika
        # Consciousness: rules control (ovca) +- panika
        # Extrovertness: merriness, energy 00 panika ?
        # Arreeableness: patience +- panika
        # Neuroticism: fear, insecurity ++ panika

        # N(0,1)
        # Panika = -ko * o - kc* c - ka* a + kn * n
        # K = [ko, kc, ka, kn]
        # Norm(K, 2) = 1
        K = [0.1, 0.1, 0.1, 0.1]
        K = K / np.linalg.norm(K, 2)
        
        # normalize traits
        mu = 0.5
        # sig = 0.1
        norm_traits = {"openness": (self.traits.openness - mu),
                       "conscientiousness": (self.traits.conscientiousness - mu),
                       "agreeableness": (self.traits.agreeableness - mu),
                       "neuroticism": (self.traits.neuroticism - mu)}
        print("traits", [self.traits.openness, self.traits.conscientiousness, self.traits.agreeableness, self.traits.neuroticism])
        print("norm_traits", norm_traits.values())
            
        self.panic_factor = -K[0] * norm_traits["openness"] - K[1] * norm_traits["conscientiousness"] - \
            K[2] * norm_traits["agreeableness"] + K[3] * norm_traits["neuroticism"] + 0.5
        
        # distributed N(0.5,0.1) same as other features
            
        print(f"Agent {self.id} panic factor: {self.panic_factor}")

    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, Agent):
            return self.id == other.id and self.position == other.position
        return False

    # def _str_(self):
    #     return f"Agent {self.id} at {self.position}, distance preference: {self.distance_preference}, velocity preference: {self.velocity_preference}"

    def __repr__(self):
        return f"Agent {self.id} at {self.position}, distance preference: {self.distance_preference}, velocity preference: {self.velocity_preference}"

    def __hash__(self):
        return hash(str(self.id) + str(self.position.x) + str(self.position.y))

    # Emotion preferences (if one large, the other small)
    def calculate_init_distance_preference(self):
        """Calculate the distance preference (one of emotion preferences)

        Returns:
            float: distance preference
        """

        # inverse relation between distance and openness,extroversion
        # positive relation between distance and agreeableness
        fO = 1 - self.traits.openness if 0 <= self.traits.openness and self.traits.openness < 0.5 else 0
        fE = 1 - self.traits.extroversion if 0 <= self.traits.extroversion and self.traits.extroversion < 0.5 else 0
        fA = 2 * self.traits.agreeableness - 1 if 0.5 <= self.traits.agreeableness else 0
        Pd = fO + fE + fA
        return Pd

    def calculate_init_velocity_preference(self):
        """Calculate the velocity preference (one of emotion preferences)

        Returns:
            float: velocity preference
        """

        # positive relation between distance and conscientiousness
        # inverse relation between distance and extroversion,neuroticism
        fC = 1 - self.traits.conscientiousness if 0 <= self.traits.openness and self.traits.openness < 0.5 else 0
        fE = self.traits.extroversion if 0.5 <= self.traits.extroversion else 0
        fN = 2 * self.traits.neuroticism - 1 if 0.5 <= self.traits.neuroticism else 0
        Pv = fC + fE + fN
        return Pv

    @functools.lru_cache(maxsize=5000)
    def d_xy(self, other: 'Agent'):
        """Calculate the positional difference between two agents
        """
        return (self.position - other.position).norm()

    @functools.lru_cache(maxsize=5000)
    def d_ori(self, other: 'Agent'):
        """Calculate the angle difference between two agents
        """
        return np.abs(np.arctan2(self.velocity.y, self.velocity.x)-np.arctan2(other.velocity.y, other.velocity.x))

    @functools.lru_cache(maxsize=5000)
    def relationship(self, other: 'Agent', cut_xy=50, cut_ori=np.pi / 3):
        """Are the agents in a collective relationship?

        Args:
            cut_xy (float, optional): Threshold for positional difference. Defaults to 50.
            cut_ori (float, optional): Threshold for angle difference. Defaults to np.pi/3.

        Returns:
            int: 1 if they are in a collective relationship, 0 otherwise
        """
        if self == other:
            return 0

        d_xy = (self.position - other.position).norm()

        # this definition differs from the article
        d_ori = np.abs(np.arctan2(self.velocity.y, self.velocity.x) -
                       np.arctan2(other.velocity.y, other.velocity.x))

        theta = np.exp(-(d_ori/cut_ori)**2) if d_ori >= cut_ori else 1 + \
            np.exp(-(d_ori/cut_ori)**2)

        share_same_goal = 1 if self.destination == other.destination else 0
        # TODO check if this is philosophically correct (is destination goal?, what if destinations are very close?)

        lam = 1 if share_same_goal else 0
        Wij = 1 if d_xy <= (cut_xy * theta) or lam == 1 else 0

        # return d_xy, d_ori, Wij
        return Wij
