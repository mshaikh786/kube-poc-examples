import os
import kfp
from kfp import compiler
from kfp import dsl

from .my_helper_module import get_model,compile_and_train, get_data_loaders
