import random
from typing import Optional
from uuid import uuid4

from graphviz import Digraph


class VisNode:
    def __init__(self, name: Optional[str] = None, color="white"):
        if name is None:
            name = str(uuid4())[-8:]
        self.name = name
        self.color = color
        self.children = []

    def add_child(self, child: "VisNode"):
        self.children.append(child)


def visualize_tree(filename: str, root_node: VisNode):
    dot = Digraph(comment="Tree", engine="dot")
    dot.attr("node", style="filled", fontcolor="black")
    dot.attr("edge", color="gray")

    def add_edges(node: VisNode):
        dot.node(node.name, fillcolor=node.color)
        for child in node.children:
            dot.edge(node.name, child.name)
            add_edges(child)

    add_edges(root_node)
    output_path = dot.render(filename, format="pdf", cleanup=True)
    print(f"Tree rendered and saved to {output_path}")


if __name__ == "__main__":

    def random_color():
        return "red" if random.choice([True, False]) else "white"

    node = VisNode("node_0", color="white")
    nodes = [node]
    for i in range(1, 100):
        new_node = VisNode(color=random_color())
        parent = random.choice(nodes)
        parent.add_child(new_node)
        nodes.append(new_node)

    visualize_tree("tree", node)
    print("Visualization complete. Open tree.pdf to view the result.")
