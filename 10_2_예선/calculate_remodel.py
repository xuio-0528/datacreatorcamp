import pandas as pd
import numpy as np

def calculate_remodel(train, test):
    after_renovation = []

    for yr_built, yr_renovated,year in zip(train['yr_built'].values, train['yr_renovated'].values, train['year'].values):
        if yr_renovated == 0: 
            after_renovation.append(year - yr_built)
        else: 
            after_renovation.append(year - yr_renovated)

    train['after_renovation'] = after_renovation

    after_renovation = []

    for yr_built, yr_renovated,year in zip(test['yr_built'].values, test['yr_renovated'].values, test['year'].values):
        if yr_renovated == 0: 
            after_renovation.append(year - yr_built)
        else: 
            after_renovation.append(year - yr_renovated)

    test['after_renovation'] = after_renovation
    return train, test