from kge.model import KgeModel
from kge.util.io import load_checkpoint

def load_model_and_dataset(checkpoint_path):
    checkpoint = load_checkpoint(checkpoint_path)
    model = KgeModel.create_from(checkpoint)
    return model, model.dataset