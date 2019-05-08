from pycorenlp import StanfordCoreNLP
import json
import googlemaps
from googlemaps import geocoding
import LocationExtraction.LocationExtracter as loc_ext
nlp = StanfordCoreNLP('http://localhost:9000')

# text = '''# Maoists trigger blast in BJP election office in JharkhandSuspected Maoists on Friday triggered a blast in an election office of the BJP at Hariharganj area of Palamu district, knocking down a part of its wall, a police officer said."Preliminary investigation suggests that armed rebels came on motorcycles around midnight, planted a bomb inside the party's office and triggered the blast," Chhatarpur Sub-divisional Police Officer Shambhu Kumar Singh said.Nobody was present inside the election office during the incident, he said.'''
# result = nlp.annotate(text,
#                    properties={
#                        'annotators': 'ner',
#                        'outputFormat': 'json',
#                        'timeout': 5000,
#                    })
#
# #result_json = json.load(result)
# print(result["sentences"])


def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)


def get_date(text):
    result = nlp.annotate(text,
                          properties={
                              'annotators': 'ner',
                              'outputFormat': 'json',
                              'timeout': 5000,
                          })
    print(result)
    try:
        for sentence in result["sentences"]:
            print(sentence)
            for entity in sentence["entitymentions"]:
                if entity['ner'] == 'DATE' and hasNumbers(entity['text']):
                    return entity['text']
    except:
        print("error")
    print("-----------------------------------------")
    return "-"


def get_location(text):
    result = nlp.annotate(text,
                          properties={
                              'annotators': 'ner',
                              'outputFormat': 'json',
                              'timeout': 5000,
                          })
    print(result)
    loc = []
    try:
        for sentence in result["sentences"]:
            print(sentence)
            for entity in sentence["entitymentions"]:
                if entity['ner'] == 'CITY' or entity['ner'] == 'LOCATION':
                    loc.append(entity['text'])
    except:
        print("error")
    print("-----------------------------------------")
    gmaps = googlemaps.Client(key='AIzaSyBqyrEuh-xeILGxddqJKBttj4uZ-NOMIic')
    result = loc_ext.get_location(gmaps, loc)
    return result


def get_org_per(text):
    org = []
    per = []
    result = nlp.annotate(text,
                          properties={
                              'annotators': 'ner',
                              'outputFormat': 'json',
                              'timeout': 5000,
                          })
    print(result)
    try:
        for sentence in result["sentences"]:
            print(sentence)
            for entity in sentence["entitymentions"]:
                if entity['ner'] == 'PERSON' and len(entity['text']) > 3 and entity['text'] not in per:
                    per.append(entity['text'])
                if entity['ner'] == 'ORGANIZATION' and entity['text'] not in org and len(entity['text']) < 30:
                    org.append(entity['text'])
    except:
        print("error")
    print("-----------------------------------------")
    per_str = ', '.join(per)
    print("persons String" + per_str)
    org_str = ', '.join(org)
    print("org String" + org_str)

    return org_str, per_str

# res, res1 = get_org_per('''# Maoists trigger blast in BJP election office in JharkhandSuspected Maoists on Friday triggered a blast in an election office of the BJP at Hariharganj area of Palamu district, knocking down a part of its wall, a police officer said."Preliminary investigation suggests that armed rebels came on motorcycles around midnight, planted a bomb inside the party's office and triggered the blast," Chhatarpur Sub-divisional Police Officer Shambhu Kumar Singh said.Nobody was present inside the election office during the incident, he said.
# ''')
# print(res)
# print(res1)




