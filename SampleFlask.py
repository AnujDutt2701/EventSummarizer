import json
from flask_cors import CORS
from flask import Flask, jsonify
from flask import request
import Classification.SpaCyClassifier as classification
import Classification.sub_event_classifier.sub_event_classifier as sub_classifier
import LocationExtraction.LocationExtracter as locationExtractor
import DataPreparation as dp
import numpy as np
import NamedEntityRecognition.EntityRelationshipExtraction as ne
import Summarizer.summarizer as summarizer
import Article_Parser as parser
import pandas as pd
from sklearn import preprocessing
import seaborn as sb
import StanfordNER.StanfordNER as stn_ner
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

@app.route('/summary/', methods=["POST"])
def summary():
    url = request.form["url"]
    article = parser.get_article(url)["content"]
    all_summary = summarizer.summarize(article, False)
    identified_locations, lat, long = stn_ner.get_location(article)
    print("Identified Locations are")
    date = stn_ner.get_date(article)
    org, per = stn_ner.get_org_per(article)
    print(identified_locations)
    final_summary = ' '.join(all_summary[0])
    final_summary = final_summary.replace('\n', ' ')
    # dictToReturn = {'article': article, 'summary': final_summary}
    dictToReturn = {'article': article, 'summary': final_summary,
                    'location': ', '.join(identified_locations), 'lat': lat, 'long':long,
                    'date': date, 'organizations': org, 'persons': per}
    return jsonify(dictToReturn)


@app.route('/classify/', methods=["POST"])
def classifier():
    article_summary = request.form["summary"]
    event_type = classification.classify(article_summary, 'Classification/spacy_classifier_new', False)
    print('------ The article is classified as ---------')
    print(event_type)
    # row.append(event_type)
    print('------ The article''s sub event is ---------')
    sub_event_type = sub_classifier.classify(article_summary,
                                             'Classification/sub_event_classifier/spacy_subevent_classifier')
    print(sub_event_type)
    dictToReturn = {'mainCategory': event_type, 'subCategory': sub_event_type}
    return jsonify(dictToReturn)


@app.route('/location/', methods=["POST"])
def location_extraction():
    article_summary = request.form["summary"]
    identified_locations, lat, long = locationExtractor.get_related_admins(article_summary)
    print("Identified Locations are")
    print(identified_locations)
    dictToReturn = {'location': identified_locations, 'lat': lat, 'long':long}
    return jsonify(dictToReturn)


@app.route('/relation/', methods=["POST"])
def relationship_extraction():
    article_summary = request.form["summary"]
    serve = request.form["serve"]
    print(serve)
    relation, related_entities, dep_html, ent_html = ne.get_relationship(article_summary, printEntites=True, serve=True)
    print(relation)
    actors = related_entities.split(',')
    actor1 = actors[0]
    actor2 = ' '.join(actors[1:])
    # event_type = classification.classify(article_summary, 'Classification/spacy_classifier_new', False)
    # if event_type == 'Violence Against Civilians':
    #     actor2 = 'Civilian'
    # if event_type == 'Protest/Riot':
    #     actor1 = 'Protesters'
    dictToReturn = {'actor1': actor1, 'actor2': actor2,'relation': relation, 'dep_html': dep_html, "ent_html": ent_html}
    return jsonify(dictToReturn)


@app.route('/entities/', methods=["POST"])
def entities():
    article_summary = request.form["summary"]
    return json.dumps({"entities": ne.get_entites(article_summary)})
    # return jsonify(ne.get_entites(article_summary))


@app.route('/comparision/', methods=["POST"])
def compared_results():
    acled_excel =  pd.read_csv("Election_demo.csv")
    system_excel = pd.read_csv("output_demo.csv")
    response = []

    for acled, system in zip(acled_excel.index, system_excel.index):
        try:
            json = {'acled_date': acled_excel["event_date"][acled], 'acled_summary': acled_excel["notes"][acled],
                    'acled_actor1': acled_excel["actor1"][acled], 'acled_actor2': acled_excel["actor2"][acled],
                    'acled_location': acled_excel["region"][acled] + ", " + acled_excel["country"][acled]+ ", "
                                      + acled_excel["admin1"][acled] + ", " +
                                      acled_excel["admin2"][acled] + ", " + acled_excel["admin3"][acled],
                    'acled_latitude': acled_excel["latitude"][acled],
                    'acled_longitude': acled_excel["longitude"][acled],
                    'acled_event_type': acled_excel["event_type"][acled], 'acled_sub_event_type': acled_excel["sub_event_type"][acled],

                    'system_date': system_excel["event_date"][system], 'system_summary': system_excel["notes"][system],
                    'system_actor1': system_excel["actor1"][system], 'system_actor2': system_excel["actor2"][system],
                    'system_location': system_excel["region"][system] + ", " + system_excel["country"][system] + ", " +
                                       system_excel["admin1"][system] + ", " +
                                       system_excel["admin2"][system] + ", " + system_excel["admin3"][system],
                    'system_latitude': system_excel["latitude"][system],
                    'system_longitude': system_excel["longitude"][system],
                    'system_event_type': system_excel["event_type"][system],
                    'system_sub_event_type': system_excel["sub_event_type"][system],
                    'persons': system_excel["persons"][system],
                    'organizations': system_excel["organizations"][system],
                    'url':system_excel["url"][system]
                    }
            response.append(json)
        except:
            print('error')

    print(response)
    return jsonify(response)


# compared_results()



