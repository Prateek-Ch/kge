from kge.model import KgeModel
from kge.util.io import load_checkpoint

class CustomDistribution:
    def __init__(self):
        self.checkpoint_path = 'local/experiments/20241021-193745-wnrr-rescal/checkpoint_best.pt'
        self.checkpoint = load_checkpoint(self.checkpoint_path)
        self.model = KgeModel.create_from(self.checkpoint)
    
    def current_distribution(self):
        print(self.model)



if __name__ == "__main__":
    custom_distribution = CustomDistribution()
    custom_distribution.current_distribution()