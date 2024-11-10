from helper_classes import Pair, Ocean
import numpy as np

class Agent:
    """
    Agent class: personality traits, emotion preference, emotion contagion
    """
    def __init__(self, id: int):
        self.id = id
        self.source : Pair = Pair(0, 0)
        self.destination : Pair = Pair(0, 0)
        
        # motion features
        self.velocity : Pair = Pair(0, 0)
        self.position : Pair = Pair(0, 0)
        
        self.traits : Ocean = Ocean.empty()
    
    # Emotion preferences (if one large, the other small)
    def calculate_distance_preference(self):
        """Calculate the distance preference (one of emotion preferences)
        
        Returns:
            float: distance preference
        """
        
        # inverse relation between distance and openness,extroversion
        # positive relation between distance and agreeableness
        fO = 1 - self.traits.openness if 0 <= self.traits.openness and self.traits.openness < 0.5 else 0
        fE = 1 - self.traits.extroversion if 0 <= self.traits.extroversion and self.traits.extroversion < 0.5 else 0
        fA = 2 * self.traits.agreeableness -1 if 0.5 <= self.traits.agreeableness else 0
        Pd = fO + fE + fA
        return Pd
    
    def calculate_velocity_preference(self):
        """Calculate the velocity preference (one of emotion preferences)
        
        Returns:
            float: velocity preference
        """
        
        # positive relation between distance and conscientiousness
        # inverse relation between distance and extroversion,neuroticism
        fC = 1 - self.traits.conscientiousness if 0 <= self.traits.openness and self.traits.openness < 0.5 else 0
        fE = self.traits.extroversion if 0.5 <= self.traits.extroversion else 0
        fN = 2 * self.traits.neuroticism -1 if 0.5 <= self.traits.neuroticism else 0
        Pv = fC + fE + fN
        return Pv
    
    def relationship(self, other : 'Agent', cut_xy = 50, cut_ori = np.pi / 3):
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
        
        # this definition deffers from the article
        # I believe this is better, than abs(arcos(vel_i) - arcos(vel_j))
        # what is arcos(<vector>) anyway?
        # if the difference in angle is for example 359 degrees, would not that be falsely very different (diff should be 1 degree in this case)
        d_ori = np.arccos((self.velocity * other.velocity) / (self.velocity.norm() * other.velocity.norm()))
        
        theta = np.exp(-(d_ori/cut_ori)**2) if d_ori >= cut_ori else 1 + np.exp(-(d_ori/cut_ori)**2)
        share_same_goal = False # TODO how do w get this information?
        lam = 1 if share_same_goal else 0
        Wij = 1 if d_xy <= (cut_xy * theta) or lam == 1 else 0
        
        return Wij
        