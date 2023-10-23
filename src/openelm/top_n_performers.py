from openelm.environments import BaseEnvironment


class TopNPerformers:
    """
    This algorithm grabs the top n performers from the list of
    candidates and returns them as the top n candidates.
    """

    def __init__(
        self,
        env,
        n: int = 2,
    ):
        """
        Args:
            n (int): The number of candidates to return. By default, this is 2.
        """
        self.n: int = n
        self.env: BaseEnvironment = env

    def search(self, candidates):
        """
        TODO: I am not totally adept to this implementation.
        Sort the candidates by fitness and return the top n.

        Args:
            candidates: A list of candidates to select from.
        Returns:
            A list of the top n candidates.
        """
        candidates.sort(key=lambda x: x.fitness, reverse=True)
        return candidates[: self.n]
