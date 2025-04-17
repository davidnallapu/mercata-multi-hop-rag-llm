# -*- coding: utf-8 -*-

# Author: Daniel Yang <daniel.yj.yang@gmail.com>
#
# License: BSD-3-Clause


from typing import Union
from pathlib import Path
import pandas as pd
import webbrowser


class Edge(dict):
  def __init__(self, source_node: 'Node' = None, to_node: 'Node' = None, weight: float = 1.0, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self['source_node'] = source_node
    self['to_node'] = to_node
    self['weight'] = weight


class Node(dict):
  def __init__(self, node_id: Union[float, int, str] = None, label: str = '', shape: str = 'circle', *args, **kwargs):
    super().__init__(*args, **kwargs)
    self['node_id'] = node_id
    self['label'] = label
    self['shape'] = shape
    self['edges'] = dict() # key: to_node_id, value: Edge
  
  def add_edge(self, to_node: 'Node' = None):
    if to_node['node_id'] not in self['edges']:
      self['edges'][to_node['node_id']] = Edge(source_node=self, to_node=to_node)


class network(object):
    def __init__(self, title="Interactive Network Analysis", width: str = '1024px', height: str = '768px', *args, **kwargs) -> None:
      super().__init__(*args, **kwargs)
      self.title = title
      self.width = width
      self.height = height
      self.nodes = dict() # key: node_id, value = Node object

    def add_node(self, node_id: Union[float, int, str] = None, label: str = ''):
      if node_id not in self.nodes:
        self.nodes[node_id] = Node(node_id = node_id, label = label)

    def add_edge(self, source_node_id: Union[float, int, str] = None, to_node_id: Union[float, int, str] = None):
      if source_node_id in self.nodes and to_node_id in self.nodes:
        self.nodes[source_node_id].add_edge(to_node = self.nodes[to_node_id])
      else:
        print('Error: the nodes are not created yet')

    def add_df(self, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> None:
      for idx, row in nodes_df.iterrows():
        self.add_node(node_id = row['node_id'], label = row['node_id'])
      for idx, row in edges_df.iterrows():
        self.add_edge(source_node_id=row['source_node_id'], to_node_id=row['to_node_id'])
      return self

    def clone_graph(self):
      new_graph = network()
      original_to_clone_mapping = {} # key: id() of original node, value: cloned node
      def clone_node(node: 'Node' = None) -> 'Node':
        nonlocal original_to_clone_mapping, new_graph
        if not node:
          return None
        if id(node) in original_to_clone_mapping:
          return original_to_clone_mapping[id(node)]
        cloned_node = Node(node_id=node['node_id'], label=node['label'], shape=node['shape'])
        new_graph.nodes[cloned_node['node_id']] = cloned_node
        original_to_clone_mapping[id(node)] = cloned_node
        if node['edges']:
          for to_node_id in node['edges']:
            cloned_node.add_edge(to_node=clone_node(self.nodes[to_node_id]))
        return cloned_node
      if self.nodes:
        clone_node(list(self.nodes.values())[0])
      return new_graph

    def _generate_html(self) -> None:
      """
      see also: https://visjs.org/
      """
      self.html = f"""<!DOCTYPE HTML>
<html>
<head>
  <title>{self.title}</title>
  <style type="text/css">
    body, html {{
      font-family: sans-serif;
    }}
    #mynetwork {{
        width: {self.width};
        height: {self.height};
        border: 1px solid lightgray;
    }}
  </style>
<script type="text/javascript" src="https://unpkg.com/vis-data@7.1.2/peer/umd/vis-data.min.js"></script>
<script type="text/javascript" src="https://unpkg.com/vis-network@9.1.0/peer/umd/vis-network.min.js"></script>
<link rel="stylesheet" type="text/css" href="https://unpkg.com/vis-network/styles/vis-network.min.css" />
<!-- Include other packages like Vis Timeline or Vis Graph3D here. -->
<div id="mynetwork"></div>
<script type="text/javascript">
  // create an array with nodes
  var nodes = new vis.DataSet(["""
      node_id_to_id_mapping = {}
      idx = 1
      for node_id in self.nodes:
        node_id_to_id_mapping[node_id] = idx
        node_id_part = f"id: {idx}, "
        idx += 1
        label_part = f"label: \"{self.nodes[node_id]['label']}\""
        self.html += f"""
    {{ {node_id_part}{label_part} }},"""
      self.html += f"""
  ]);
  // create an array with edges
  var edges = new vis.DataSet(["""
      for node_id in self.nodes:
        from_part = f"from: {node_id_to_id_mapping[node_id]}, "
        for to_node_id in self.nodes[node_id]['edges']:
          to_part = f"to: {node_id_to_id_mapping[to_node_id]}"
          self.html += f"""
    {{ {from_part}{to_part} }},"""
      self.html += f"""
  ]);
  // create a network
  var container = document.getElementById("mynetwork");
  var data = {{
    nodes: nodes,
    edges: edges
  }};
  var options = {{}};
  var network = new vis.Network(container, data, options);
</script>
</body>
</html>"""

    def write_html(self, filename: Union[Path, str]) -> None:
      self._generate_html()
      output_path = filename
      if not isinstance(output_path, Path):
        output_path = Path(output_path)
      output_path.write_text(self.html)

    def show(self, filename: Union[Path, str]) -> None:
      output_path = filename
      if not isinstance(output_path, Path):
        output_path = Path(output_path)
      self.write_html(filename=output_path)
      webbrowser.open(output_path.resolve().as_uri(), new = 2)



