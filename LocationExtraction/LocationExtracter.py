import googlemaps
from googlemaps import geocoding
import NamedEntityRecognition.NETagging as ner


def get_related_admins(document):
    gmaps = googlemaps.Client(key='AIzaSyBqyrEuh-xeILGxddqJKBttj4uZ-NOMIic')
    spacy_nlp = ner.get_ner()
    doc = spacy_nlp(document)
    locations_tags = ['NORP', 'GPE', 'LOC']
    all_entities = [(ent.text, ent.label_) for ent in doc.ents]
    locations = [loc for loc, tag in all_entities if locations_tags.__contains__(tag)]

    # print(all_entities)
    # print(locations)
    return get_location(gmaps, locations)


def get_location(gmaps, locations):
    identified_locations = []
    lat = ''
    long = ''
    for location in locations:
        if location in identified_locations or not location.isalpha():
            continue
        address = geocoding.geocode(gmaps, location)
        admins = []
        if len(address) == 0: continue
        for i in range(len(address[0]['address_components'])):
            loc = address[0]['address_components'][i]['long_name']
            postal_code_type = address[0]['address_components'][i]['types'][0]
            if postal_code_type == 'postal_code':
                continue
            # if loc not in identified_locations:
            admins.append(loc)
        # print(admins)
        if 'India' not in admins:
            continue
        if len(identified_locations) < len(admins):
            lat = address[0]['geometry']['location']['lat']
            long = address[0]['geometry']['location']['lng']
            identified_locations = admins
        #identified_locations.extend(admins)
    identified_locations.reverse()
    if not identified_locations:
        identified_locations.append(locations[0] if locations else "")

    return identified_locations, lat, long

#
# result = get_location(gmaps = googlemaps.Client(key='AIzaSyBqyrEuh-xeILGxddqJKBttj4uZ-NOMIic'), locations=['jalpura'])
# print(result)
