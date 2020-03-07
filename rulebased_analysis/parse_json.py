import json
from ast import literal_eval

def parse_json(file):

    with open(file) as json_file:
        json_data = json.load(json_file)
        json_data = json_data["facilities"]
        for facility in json_data:
            for point_group in facility['hp'], facility['rp']:
                for point in point_group:
                    for idx, pixel in enumerate(point):
                        point[idx] = literal_eval(pixel)
        
    return json_data

if __name__ == "__main__":
    parse_json('./json_rb/frame00012.json')
    pass