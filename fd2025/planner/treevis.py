import random
from typing import List, Optional
from uuid import uuid4

from graphviz import Digraph


class VisNode:
    def __init__(self, name: Optional[str] = None, color="white"):
        self.name = name or str(uuid4())[-8:]
        self.color = color
        self.children: List["VisNode"] = []

    def add_child(self, child: "VisNode"):
        self.children.append(child)


def visualize_tree(
    filename: str,
    root_node: VisNode,
    n_threshold: int = 5,
    engine: str = "dot",
):
    dot = Digraph(comment="Tree", engine=engine)
    dot.attr(
        "node",
        style="filled",
        fontcolor="black",
        shape="circle",
        fixedsize="true",
        width="0.2",
        height="0.2",
        label="",
    )
    dot.attr("edge", color="black")
    dot.attr("graph", repulsiveforce="8.0", K="0.1")

    def add_edges(node: VisNode):
        dot.node(node.name, fillcolor=node.color)

        leaf_children = [c for c in node.children if not c.children]
        branch_children = [c for c in node.children if c.children]

        if len(leaf_children) > n_threshold:
            for child in leaf_children[:n_threshold]:
                dot.edge(node.name, child.name)
                dot.node(child.name, fillcolor=child.color)

            remain = len(leaf_children) - n_threshold
            group_name = f"{node.name}_plus_{remain}"
            group_label = f"+{remain}"
            group_color = leaf_children[0].color  # NOTE: all leaves should have the same color
            dot.node(
                group_name,
                label=group_label,
                shape="box",
                fillcolor=group_color,
                fontcolor="black",
                fixedsize="false",
                width="0.6",
                height="0.25",
                margin="0.04,0.02",
            )
            dot.edge(node.name, group_name)
        else:
            for child in leaf_children:
                dot.edge(node.name, child.name)
                dot.node(child.name, fillcolor=child.color)

        for child in branch_children:
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
