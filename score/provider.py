from abc import abstractmethod, ABCMeta

from data.model import *


class BaseScoreDataProvider(metaclass=ABCMeta):

    def __init__(self, weights: ScoreModelWeight, config: NetworkConfig,
                 connection: sqlite3.Connection):
        self.weights = weights
        self.config = config
        self.connection = connection

    @abstractmethod
    def provide_user_data(self, user_key: str, epoch, ignore_less=True):
        """
        get the training data (scores) for a user from database
        @param user_key: the target user, in the format of {user_id}-{game_mode}-{variant}
        @param epoch: training epoch
        @param ignore_less: when lacking training data, return None
        @return: (scores, beatmap_id_array, mod_id_array, weights)
            - scores: shape = [N, ]
            - beatmap_id_array: shape = [N, ], the embedding id of weights.beatmap_embedding
            - mod_id_array: shape = [N, ], the embedding id of weights.mod_embedding
            - weights: shape = [N, ], training weight for each data sample.
        """
        pass

    @abstractmethod
    def provide_beatmap_data(self, beatmap_key: str, epoch, ignore_less=True):
        """
        get the training data (scores) for a beatmap from database
        @param beatmap_key: the target beatmap
        @param epoch: training epoch
        @param ignore_less: when lacking training data, return None
        @return: (scores, user_id_array, mod_id_array, weights)
            - scores: shape = [N, ]
            - user_id_array: shape = [N, ], the embedding id of weights.user_embedding
            - mod_id_array: shape = [N, ], the embedding id of weights.mod_embedding
            - weights: shape = [N, ], training weight for each data sample.
        """
        pass

    @abstractmethod
    def provide_mod_data(self, mod_key: str, epoch, ignore_less=True):
        """
        get the training data (scores) for a mod from database.
        @param mod_key: the target mod. For example, DT.
        @param epoch: training epoch
        @param ignore_less: when lacking training data, return None
        @return: (scores, user_id_array, mod_id_array, weights)
            - scores: shape = [N, ]
            - user_id_array: shape = [N, ], the embedding id of weights.user_embedding
            - beatmap_id_array: shape = [N, ], the embedding id of weights.beatmap_embedding
            - weights: shape = [N, ], training weight for each data sample.
        """
        pass

