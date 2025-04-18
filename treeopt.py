import networkx as nx
import matplotlib.pyplot as plt

def graph_to_latex(G : nx.Graph, edge_labels, node_opts = None, edge_opts = None, scale = 1, tikz_file_name = "tikz.tex"):
        tikz_opts = "scale="+str(scale) + ", ultra thick, node_style/.style={circle,draw=blue,fill=blue!20!,scale=0.6,font=\\tiny},terminal_style/.style={circle,draw=red,fill=white,scale=0.6,font=\\tiny},edge_style/.style={draw=black, thick,font=\\small},selected_edge_style/.style={draw=red, thick, font=\\small}"
        if edge_opts is None:
            edge_opts = {e: "edge_style" for e in G.edges}
        edge_label_opts = {e: "near start" for e in G.edges}
        if node_opts is None:
            node_opts = {i: "node_style" for i in G.nodes}
        latex_code = nx.to_latex(G,
                        tikz_options = tikz_opts,
                        node_options=node_opts, 
                        edge_options=edge_opts, 
                        edge_label=edge_labels,
                        edge_label_options=edge_label_opts, 
                        as_document = False, caption="test", 
                        latex_label="fig:soln")
        with open(tikz_file_name, "w") as f:
            f.write(latex_code)

def get_digraph(G : nx.Graph):
    diGraph = nx.DiGraph()
    for i in G.nodes():
        diGraph.add_node(i)

    for u, v in G.edges():
        diGraph.add_edge(u, v, weight=G[u][v]['weight'])
        diGraph.add_edge(v, u, weight=G[u][v]['weight'])
    return diGraph

def display_steiner_solution(G : nx.Graph, steiner_tree, node_colors):
     # Set edge attributes for Steiner tree edges
    for u, v in G.edges():
        if (u, v) in steiner_tree.edges():
            G[u][v]['color'] = 'red'
            G[u][v]['width'] = 3.0
        else:
            G[u][v]['color'] = 'blue'
            G[u][v]['width'] = 1.0


    edge_colors = ['red' if G[u][v].get('color') == 'red' else 'blue' for u, v in G.edges()]
    widths = [G[u][v]['width'] for u, v in G.edges()]

    pos = {i: G.nodes[i]['pos'] for i in G.nodes()}
    # Draw the graph with Steiner tree edges in red and thicker
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, width=widths, node_size=500)

    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.show()

def get_edge_style(e, E):
    if e in E:
        return "selected_edge_style"
    else:
        return "edge_style"