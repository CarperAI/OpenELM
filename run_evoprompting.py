"""
    This is an implementation of evoprompting. It is a genetic algorithm that
    uses a neural network to generate new candidates. It is based on the
    following paper:
"""

import hydra
from omegaconf import OmegaConf

from openelm.configs import ImageEnvConfig
from openelm.environments import ImageOptim
from openelm.mutation_model import DiffModel, MutationModel, PromptModel


class EvoPrompting:
    def __init__(self, config: ImageEnvConfig) -> None:
        """
            Use Evoprompting to create a new candidate.
        """
        self.config: ImageEnvConfig = config
        print(self.config)

        # Model
        if self.config.model.model_name == 'prompt':
            self.mutation_model: MutationModel = PromptModel(self.config.model)
        elif self.config.model.model_name == 'diff':
            self.mutation_model: MutationModel = DiffModel(self.config.model)

    def run(self):
        """
            Do the Evoprompting algorithm.
        """
        env = ImageOptim(mutation_model=self.mutation_model, config=self.config)
        # p = 2
        # getTop = TopNPerformers(p)
        T = 10
        t = 0
        P = env.random()
        G = []
        evaluated_metric = []

        while t < T:
            G = []
            C = env.mutate(P)
            C_evaled = self.filter_and_eval(C)
            G.append(C_evaled)
            print("iteration: ", t)
            if t < T - 1:

                P = G[-2:]
                P = [item[0] for sublist in P for item in sublist]
                current_metrics = [item[1] for sublist in G[-2:] for item in sublist]
                evaluated_metric.append(current_metrics)  # Append the fitness
            print("G: ", G)
            print("P: ", P)
            print("Evaluated: ", evaluated_metric)
            print("-----------------")
            # gc.collect() # do something about garbage collection
            t += 1

        # Print the code that is the population
        # inside of the population that we got
        # from the last iteration
        print("Population: ", P[0].program_str)

    def filter_and_eval(self, candidates: list) -> list:
        """
            Filter the candidates and evaluate them.
            Algorithm 3. in the paper
        """
        env = ImageOptim(mutation_model=self.mutation_model, config=self.config)

        C_evaled = []
        # TODO: Check why why you loose candidates, no matter
        # what the filter_alpha is.
        for c in candidates:
            if True:  # self.config.model.filter_alpha: # TODO: Filter by alpha
                fitness = env.fitness(c)
                tuple = (c, fitness)
                C_evaled.append(tuple)

        C_evaled.sort(key=lambda x: x[1], reverse=True)

        return C_evaled


@hydra.main(
    config_name="image_evolution",
    version_base="1.2",
)
def main(config):
    print("----------------- Config ---------------")
    print(OmegaConf.to_yaml(config))
    print("-----------------  End -----------------")

    evoPrompting = EvoPrompting(config)
    evoPrompting.run()


if __name__ == "__main__":
    main()
