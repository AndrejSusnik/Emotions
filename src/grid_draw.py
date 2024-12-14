import networkx as nx
import matplotlib.pyplot as plt

# Your grid data
grid = [
    [((1, 0), 5), ((1, 1), 4), ((1, 2), 3), ((1, 3), 2), ((1, 4), 1), ((1, 5), 0)],
    [((1, 1), 4), ((1, 2), 3), ((1, 3), 2), ((1, 4), 1), ((1, 5), 0), None],
    [((2, 1), 4), ((2, 2), 3), ((2, 3), 2), ((2, 4), 1), ((2, 5), 0), None],
    [((2, 0), 5), ((2, 1), 4), ((2, 2), 3), ((2, 3), 2), ((2, 4), 1), ((2, 5), 0)]
]

G = nx.DiGraph()
color_map = {}
# Add nodes and edges
for i, row in enumerate(grid):
    for j, cell in enumerate(row):
        G.add_node((i, j), pos=(j, i))
        color_map[(i, j)] = 'lightblue' if cell  else "yellow"
        
for i, row in enumerate(grid):
    for j, cell in enumerate(row):
        if cell is not None:
            G.add_edge((i, j), cell[0])
            
color_list = [color_map[node] for node in G.nodes()] 


# Draw the graph
pos = nx.get_node_attributes(G, 'pos')
labels = nx.get_node_attributes(G, 'distance')
nx.draw(G, pos, with_labels=True, labels=labels, node_size=700, node_color = color_list, font_size=10, font_color='black')
plt.show()