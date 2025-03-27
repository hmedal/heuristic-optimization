from math import radians, sin, cos, sqrt, atan2
import networkx as nx

def get_xy_coordinates(city, df):
    row = df[df['City'] == city]
    x = row['X Coordinate'].values[0]
    y = row['Y Coordinate'].values[0]
    return x, y

def get_latlng_coordinates(city, df):
    row = df[df['City'] == city]
    lat = row['Latitude'].values[0]
    lng = row['Longitude'].values[0]
    return lat, lng

def compute_distance(city1, city2, df):
    lat1, lon1 = get_latlng_coordinates(city1, df)
    lat2, lon2 = get_latlng_coordinates(city2, df)
    
    # approximate radius of earth in miles
    R = 3958.8

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def get_airport_code(city, df):
    row = df[df['City'] == city]
    airport_code = row['Airport Code'].values[0]
    return airport_code

def get_cities_graph(df_cities, dist_min, dist_max):
    G = nx.DiGraph()

    city_names = df_cities['City'].tolist()

    for city in city_names:
        code = get_airport_code(city, df_cities)
        airport_code = get_airport_code(city, df_cities)
        #print(airport_code)
        x,y = get_xy_coordinates(city, df_cities)

        if x is not None:
            G.add_node(code, name = city, pos=(x,y))

    max_x = max([x for x, y in nx.get_node_attributes(G, 'pos').values()])
    min_x = min([x for x, y in nx.get_node_attributes(G, 'pos').values()])
    max_y = max([y for x, y in nx.get_node_attributes(G, 'pos').values()])
    min_y = min([y for x, y in nx.get_node_attributes(G, 'pos').values()])

    x_diff = max_x - min_x
    y_diff = max_y - min_y
    ratio = y_diff / x_diff
 
    for u in G.nodes():
        x, y = G.nodes[u]['pos']
        x = (x - min_x) / (max_x - min_x)
        y = ratio*((y - min_y) / (max_y - min_y))
        G.nodes[u]['pos'] = (x, y)

    # Add edges between nodes and assign weights proportional to the distance
    for u in G.nodes():
        for v in G.nodes():
            if u != v:
                x1, y1 = G.nodes[u]['pos']
                x2, y2 = G.nodes[v]['pos']
                #print(x1, y1, x2, y2)
                city1Name = G.nodes[u]['name']
                city2Name = G.nodes[v]['name']
                distance = compute_distance(city1Name, city2Name, df_cities)
                if distance < dist_max and distance > dist_min:
                    G.add_edge(u, v, weight=distance)
    return G