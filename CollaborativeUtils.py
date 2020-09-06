# Libraries import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise import KNNBasic,  KNNWithMeans, KNNBaseline, SVD
from surprise.model_selection import KFold
import time
from tqdm import tqdm
