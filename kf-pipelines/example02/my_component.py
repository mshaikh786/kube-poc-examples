import os
import kfp
from kfp import compiler
from kfp import dsl

from .my_helper_module import get_model,compile_and_train, get_data_loaders


@dsl.component(
    base_image='python:3.7',
    target_image='docker.io/mshaikh/tf-kf:v1',
    packages_to_install=['torch==1.10.0',
                         'torchvision==0.11.1'],
)
def train_model(
    dataset: Input[Dataset],
    model: Output[Model],
    num_epochs: int,
):
    with open(dataset.path) as f:
        x, y = get_data_loaders(f)
    
    untrained_model = get_model()
    
    trained_model = compile_and_train(untrained_model,
                                      epochs=2)
    
    trained_model.save(model.path)
                                      