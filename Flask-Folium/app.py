# Import necessary libraries
from flask import Flask, render_template_string
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tsp_map
from flask import request

# Create a Flask web application
app = Flask(__name__)
initial_city = 'Austin'

# Create a route to display the map
@app.route('/')
def index():
    folium_map, cost = tsp_map.generate_html_info(initial_city)
    formatted_cost = round(cost)
    map_html = folium_map._repr_html_()
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>TSP</title>
        <style>
            body {{
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                height: 100vh;
                margin: 0;
            }}
            #map {{
                width: 650px;
                height: 400px;
            }}
            button {{
                margin-top: 10px;
            }}
        </style>
    </head>
    <body>
        <div id="map">{map_html}</div>
        <p id="cost"> Cost of the TSP tour: {formatted_cost} </p>
        <label for="starting_city">Starting City:</label>
        <input type="text" id="starting_city_input" placeholder="Enter starting city" value="{initial_city}">
        <button onclick="refreshMap()">Refresh Map</button>
        <script>
            function refreshMap() {{
                const starting_city_input_val = document.getElementById('starting_city_input').value;
                fetch(`/refresh_map?starting_city_val=${{starting_city_input_val}}`)
                    .then(response => response.json())
                    .then(data => {{
                        document.getElementById('map').innerHTML = data.map_html;
                        document.getElementById('cost').innerText = `Cost of the TSP tour: ${{data.cost}}`;
                    }});
            }}
        </script>
    </body>
    </html>
"""
    return render_template_string(html_template)

# Create a route to refresh the map
@app.route('/refresh_map')
def refresh_map():
    print("args", request.args)
    starting_city_val = request.args.get('starting_city_val')
    print("starting_city_val: ", starting_city_val)
    folium_map, cost = tsp_map.generate_html_info(starting_city_val)
    response = {
        'map_html': folium_map._repr_html_(),
        'cost': round(cost)
    }
    return response

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)