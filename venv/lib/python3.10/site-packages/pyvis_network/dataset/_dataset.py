# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD-3-Clause


import pandas as pd
import importlib.resources as pkg_resources
from io import StringIO
from . import data


class dataset(object):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_as_df(self, name: str):
        if name == "Example1":
            self.nodes_df = pd.read_csv(StringIO(pkg_resources.read_text(data, 'example1_nodes.csv')), keep_default_na=False)
            self.edges_df = pd.read_csv(StringIO(pkg_resources.read_text(data, 'example1_edges.csv')), keep_default_na=False)
        elif name == "Machine_Learning":
            self.nodes_df = pd.read_csv(StringIO(pkg_resources.read_text(data, 'Machine_Learning_nodes.csv')), keep_default_na=False)
            self.edges_df = pd.read_csv(StringIO(pkg_resources.read_text(data, 'Machine_Learning_edges.csv')), keep_default_na=False)            
        return self.nodes_df, self.edges_df
        
