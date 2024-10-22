from kge.model import KgeModel
from kge.util.io import load_checkpoint

checkpoint = load_checkpoint('local/experiments/20241021-193745-wnrr-rescal/checkpoint_best.pt')
model = KgeModel.create_from(checkpoint)
print(model)