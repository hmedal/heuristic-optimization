import folium
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import util
import networkx as nx

class City():
    marker_color = 'blue'

    def __init__(self, name, lat, lon):
        self.name = name
        self.lat = lat
        self.lon = lon

    def get_popup(self):
        return self.name
    
    def add_to_map(self, folium_map):
        folium.Marker(location=(self.lat, self.lon), popup=self.get_popup(), 
                      icon=folium.Icon(color=self.marker_color)).add_to(folium_map)

def add_line(folium_map, loc1 : City, loc2 : City, color = 'blue'):
    point1 = (loc1.lat, loc1.lon)
    point2 = (loc2.lat, loc2.lon)
    folium.PolyLine(locations=[point1, point2], color=color).add_to(folium_map)

def nearest_neighbor_heuristic(G : nx.DiGraph, starting_node : str):
    visited = []
    current_node = starting_node
    visited.append(current_node)
    while len(visited) < len(G.nodes()):
        neighbors = [j for j in G.neighbors(current_node) 
                     if j not in visited]
        nearest_neighbor = min(neighbors, 
            key=lambda node: G[current_node][node]['weight'])
        current_node = nearest_neighbor
        visited.append(current_node)
    visited.append(visited[0])
    return visited

def get_cost_of_tour(G : nx.DiGraph, edges_used):
    return sum(G[u][v]['weight'] for u, v in edges_used)

def generate_html_info(starting_city = 'Austin'):
    start_coords = (39.8283, -98.5795)  # Center of the United States
    folium_map = folium.Map(location=start_coords, zoom_start=3.5, width='100%')
    df_cities = pd.read_csv('../cities.csv')
    
    cities = {row['City'] : City(row['City'], row['Latitude'], row['Longitude']) for index, row in df_cities.iterrows()} 
    for c in cities:
        cities[c].add_to_map(folium_map)

    G = util.get_cities_graph(df_cities)
    nodes_visited = nearest_neighbor_heuristic(G, starting_city)
    edges_used = list(zip(nodes_visited, nodes_visited[1:]))
    for i,j in edges_used:
        add_line(folium_map, cities[i], cities[j], 'red')
    return folium_map, get_cost_of_tour(G, edges_used)