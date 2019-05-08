import Classification.SpaCyClassifier as classification
import Classification.sub_event_classifier.sub_event_classifier as sub_classifier
import LocationExtraction.LocationExtracter as locationExtractor
import DataPreparation as dp
import StanfordNER.StanfordNER as stn_ner
import numpy as np
import NamedEntityRecognition.EntityRelationshipExtraction as ne
import Summarizer.summarizer as summarizer
import pandas as pd
from sklearn import preprocessing
import seaborn as sb
import matplotlib.pyplot as plt
import  Article_Parser as scrapper


def get_at_index(array, index):
    value = ''
    try:
        value = str(array[index])
    except IndexError:
        if len(array) == 0:
            return ""
        value = array[-1]
    return value

u'''India has lodged a protest with Pakistan over four separate incidents of
    alleged harassment of Indian High Commission officials in Islamabad and sought
    an immediate investigation into them, official sources said Thursday.'''

final_summary = '''students of a higher secondary school in Tamil Nadu’s Ambur have refused to write down their examination in the state’s native language, Tamil. Mazharul Uloom higher secondary school on 9th March witnessed widespread protests with more than 200 students reluctant to write their examination in any other language apart from Urdu. '''# To calm down the entire situation, District Educational Officer of Vellore himself had to intervene. The DEO listened to the grievances of the students and assured that their concerns would be forwarded to higher authorities. Only after DEO’s intervention, the students called-off their protest.
#
# In 2006, The Government of Tamil Nadu passed an order to give importance to Tamil language in schools across the state. The government had said that the order would be implemented gradually with every passing year.
#
# In 2018, a case was filed in the Madras High Court, against this order on behalf of the minority language speakers. Giving relief to those speaking minority languages, Madras High Court had granted exemption to students studying in linguistic minority institutions from taking compulsory Tamil examination in last year’s board exams.'''

def run():
    articles = pd.read_csv('Summarizer/Milestone3.csv',encoding="ISO-8859-1") # Election_Demo_articles.csv
    dates = []
    persons = []
    organizations = []
    urls = []
    for article, person, organization, url in zip(articles["article_text"], articles["persons"], articles["organizations"], articles["url"]):
        dates.append( stn_ner.get_date(article))
        persons.append(person)
        organizations.append(organization)
        urls.append(url)

    # for person in articles["persons"]:
    #
    #
    # for organization in articles["organizations"]:
    #     organizations.append(organization)

    all_summary = summarizer.summarize('Summarizer/Milestone3.csv', True)
    print('------ Summarized Results ---------')
    print(all_summary)
    # all_summary = [['Giving relief to those speaking\nminority languages, Madras High Court had granted exemption to students\nstudying in linguistic minority institutions from taking compulsory Tamil\nexamination in last year\x92s board exams.', 'In 2006, The Government of Tamil Nadu passed an order to give importance to\nTamil language in schools across the state.'], ['In her police complaint, one of the abducted girls \x97 a 15-year-old class X student \x97 said she and her cousin study in the same school and were befriended by the accused minor, who was apparently living away from her hometown.', 'In her police complaint, one of the abducted girls \x97 a 15-year-old class X student \x97 said she and her cousin study in the same school and were befriended by the accused minor, who was apparently living away from her hometown.'], ['A protester in front of the party headquarters said that people from all over the state belonging to several movement groups are opposing the AGP-BJP ties.', 'Mahanta had also said that the saffron party intends to bring in the Citizenship (Amendment) Bill if they return to power after the polls.']]
    output = pd.DataFrame(
        columns=['data_id', 'event_date', 'year', 'event_type', 'sub_event_type', 'actor1', 'assoc_actor_1', 'actor2',
                 'assoc_actor_2', 'region', 'country', 'admin1', 'admin2', 'admin3', 'location', 'latitude',
                 'longitude', 'notes'])
    test = []
    data_id = 0
    print(dates)
    for summary in all_summary:
        event_date = dates[data_id]
        persons_str = persons[data_id]
        organizations_str = organizations[data_id]
        url_str = urls[data_id]
        data_id += 1
        final_summary = summary
        final_summary = ' '.join(final_summary)
        final_summary = final_summary.replace('\n', ' ')
        row = []
        # test_text = 'A protester in front of the party headquarters said that people from all over the state belonging to several movement groups are opposing the AGP-BJP ties.'
        event_type = classification.classify(final_summary, 'Classification/spacy_classifier_new', False)
        print('------ The article is classified as ---------')
        print(event_type)
        row.append(event_type)
        print('------ The article''s sub event is ---------')
        sub_event_type = sub_classifier.classify(final_summary,
                                                 'Classification/sub_event_classifier/spacy_subevent_classifier')
        identified_locations, lat, long = stn_ner.get_location(final_summary)
        print("Identified Locations are")
        print(identified_locations)
        row.extend(identified_locations)
        country = ''

        row.append(lat)
        row.append(long)
        # test_text = 'A protester in front of the party headquarters said that people from all over the state belonging to several movement groups are opposing the AGP-BJP ties.'
        relation, related_entities, dep_html, ent_html = ne.get_relationship(final_summary, printEntites=True, serve=False)
        print(relation)
        actors = related_entities.split(',')
        actor1 = actors[0]
        actor2 = ' '.join(actors[1:])
        if event_type == 'Violence Against Civilians':
            actor2 = 'Civilian'
        if event_type == 'Protest/Riot':
            actor1 = 'Protesters'
        row.append(actor1)
        row.append(actor2)
        print('actor1 is: ' + actor1 + ' actor2 is:' + actor2)

        test.append(row)

        result = {'data_id': data_id, 'event_date': event_date, 'year': 2019, 'event_type': event_type,
                  'sub_event_type': sub_event_type, 'actor1': actor1, 'assoc_actor_1': '',
                  'actor2': actor2, 'assoc_actor_2': '', 'region': 'Southern Asia',
                  'country': 'India',  # get_at_index(identified_locations,0),
                  'admin1': get_at_index(identified_locations, 1),
                  'admin2': get_at_index(identified_locations, 2),
                  'admin3': get_at_index(identified_locations, 3),
                  'location': get_at_index(identified_locations, 4),
                  'latitude': lat, 'longitude': long, 'notes': final_summary,
                  'persons': persons_str, 'organizations': organizations_str, 'url': url_str}

        output = output.append(result, ignore_index=True)
    print(output)
    print(test)
    output.to_csv("output_demo_m3.csv", index=False, encoding='utf-8')


def calculate_confusion_matrix(output, testing_target):
    confusion_matrix_py = np.zeros((8,8), dtype="int")
    for i, j in zip(output, testing_target):
        confusion_matrix_py[i][j] += 1
    print(confusion_matrix_py)
    return confusion_matrix_py


def calculate_confusion_matrix_sub_event(output, testing_target):
    sub_type= ['Abduction/forced disappearance','Attack','Sexual violence','Mob violence','Peaceful protest',
               'Protest with intervention','Violent demonstration','Excessive force against protesters']
    confusion_matrix_py = np.zeros((8,8), dtype="int")
    for out, tar in zip(output, testing_target):
        i = sub_type.index(out)
        j = sub_type.index(tar)
        confusion_matrix_py[i][j] += 1
    print(confusion_matrix_py)
    return confusion_matrix_py


def create_heat_map(confusion_matrix):
    dataframe = pd.DataFrame(confusion_matrix)
    plt.figure(figsize=(10,10))
    sb.heatmap(dataframe, annot=True, fmt="g", cbar=True)
    plt.show()


def evaluate_sub_category():
    output = pd.read_csv("output.csv", encoding='utf-8')
    gold_standard = pd.read_csv("GoldStandard.csv", encoding="utf-8")
    correct = 0
    wrong = 0
    scores = []
    for note, label in zip(output['notes'], gold_standard['sub_event_type']):
        score = sub_classifier.classify(note, 'Classification/sub_event_classifier/spacy_subevent_classifier')
        scores.append(score)
        if score == label:
            correct += 1
        else:
            wrong += 1
    accuracy = correct / (correct + wrong)
    print(accuracy)
    return scores, gold_standard['sub_event_type']


def evaluate():
    encoder = preprocessing.LabelEncoder()
    output = pd.read_csv("output.csv",encoding='utf-8')
    gold_standard = pd.read_csv("GoldStandard.csv", encoding="utf-8")
    gold_standard.loc[gold_standard['event_type'].isin(['Protests', 'Riots']), 'event_type'] = 'Protest/Riot'
    gold_standard['event_type'] = encoder.fit_transform(gold_standard['event_type'])

    print(encoder.classes_)
    # df = output['notes'] + gold_standard['event_type']
    df = pd.concat([output['notes'],  gold_standard['event_type']], axis=1, sort=False)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for note, label in zip(output['notes'],gold_standard['event_type']):
        score = classification.classify(note, 'Classification/spacy_classifier_new', False)
        #score = cats['RiotsProtests'] if cats['RiotsProtests'] > cats['ViolenceAgainstCivilians'] else cats['ViolenceAgainstCivilians']
        if score == 'Protest/Riot' and label == 0:
            tp += 1.0
        elif score == 'Protest/Riot' and label == 1:
            fp += 1.0
        elif score == 'Violence Against Civilians' and label == 1:
            tn += 1
        elif score == 'Violence Against Civilians' and label == 0:
            fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn)/(tp+tn+fp+fn)
    print({"tp": tp, "fp": fp, "fn": fn,"tn":tn})
    print({"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score})
    print(accuracy)


def collect_non_elections_data():
    raw_data = pd.read_csv("Election_Demo_url.csv")
    #raw_data = pd.read_csv('Summarizer/Milestone3.csv', encoding="ISO-8859-1")
    print(raw_data["article_url"])
    df = pd.DataFrame(columns=['article_text', 'persons', 'organizations', 'url'])
    for url in raw_data["article_url"]:
        raw_content = scrapper.get_article(url)
        if raw_content:
            content = raw_content["content"]
            org, per = stn_ner.get_org_per(content)
            df = df.append({'article_text': content, 'persons': per, 'organizations': org, 'url':url}, ignore_index=True)
    print(df)
    df.to_csv("Election_Demo_articles.csv")
    print(df)


def actor_classification():
    gold = pd.read_csv('Election_Demo.csv', encoding="utf-8")
    out = pd.read_csv('output_demo.csv', encoding="utf-8")

    gold_result = zip(gold['actor1'], gold['actor2'])
    out_result = zip(out['actor1'], out['actor2'])

    avg = 0
    count = 0
    for x, y in zip(gold_result, out_result):
        count = 0
        x_values = x[0].split()
        y_values = y[0].split()
        for val in x_values:
            if val in y_values:
                count += 0.5

        avg += count

    print(avg / len(gold) * 100)

# run()
# relation, related_entities, dep_html, ent_html = ne.get_relationship('''In June, an army soldier Aurangzeb of 44 Rashtriya Rifles posted in south Kashmirs Shopian district was abducted by militants and his bullet-riddled body was found 10 kilometers away from the place of kidnapping. Soon after police came to know about abduction of the soldier, a joint police and army team was send to village and search operation was launched in the area.Officials said that Bhat was at his home at Qazipora when some unidentified gunmen abducted him from his house.''', printEntites=True, serve=True)
# evaluate()
# output, target = evaluate_sub_category()
# output = ['Mob violence', 'Mob violence', 'Mob violence', 'Violent demonstration', 'Mob violence', 'Excessive force against protesters', 'Excessive force against protesters', 'Protest with intervention', 'Sexual violence', 'Mob violence', 'Excessive force against protesters', 'Mob violence', 'Sexual violence', 'Mob violence', 'Attack', 'Sexual violence', 'Mob violence', 'Mob violence', 'Mob violence', 'Peaceful protest', 'Attack', 'Mob violence', 'Mob violence', 'Mob violence', 'Mob violence', 'Sexual violence', 'Excessive force against protesters', 'Attack', 'Mob violence', 'Sexual violence', 'Sexual violence', 'Excessive force against protesters', 'Mob violence', 'Mob violence', 'Mob violence', 'Peaceful protest', 'Abduction/forced disappearance', 'Sexual violence', 'Excessive force against protesters', 'Attack', 'Mob violence', 'Excessive force against protesters', 'Mob violence', 'Mob violence', 'Sexual violence', 'Mob violence', 'Mob violence', 'Mob violence', 'Mob violence', 'Mob violence', 'Mob violence', 'Attack', 'Mob violence', 'Protest with intervention', 'Attack', 'Mob violence', 'Peaceful protest', 'Mob violence', 'Mob violence', 'Mob violence', 'Mob violence', 'Violent demonstration', 'Sexual violence', 'Attack', 'Mob violence', 'Attack', 'Mob violence', 'Attack', 'Attack', 'Mob violence', 'Mob violence', 'Attack', 'Attack', 'Mob violence', 'Mob violence', 'Sexual violence', 'Attack', 'Mob violence', 'Attack', 'Attack', 'Mob violence', 'Attack', 'Attack', 'Mob violence']
# target = ['Violent demonstration', 'Mob violence', 'Peaceful protest', 'Peaceful protest', 'Protest with intervention', 'Peaceful protest', 'Peaceful protest', 'Protest with intervention', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Mob violence', 'Mob violence', 'Peaceful protest', 'Peaceful protest', 'Mob violence', 'Peaceful protest', 'Peaceful protest', 'Mob violence', 'Mob violence', 'Mob violence', 'Mob violence', 'Mob violence', 'Mob violence', 'Protest with intervention', 'Protest with intervention', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Mob violence', 'Violent demonstration', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Mob violence', 'Mob violence', 'Mob violence', 'Mob violence', 'Peaceful protest', 'Mob violence', 'Mob violence', 'Mob violence', 'Protest with intervention', 'Attack', 'Mob violence', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Peaceful protest', 'Mob violence', 'Peaceful protest', 'Mob violence', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Abduction/forced disappearance', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack', 'Attack']
# matrix = calculate_confusion_matrix_sub_event(output, target)
# create_heat_map(matrix)

# import NamedEntityRecognition.NETagging as ner
# from spacy import displacy
# spacy_nlp = ner.get_ner()
# options = {"bg": "#ffffff", "color": "#212121", "font": "Roboto"}
# doc = spacy_nlp("The cat sat on the mat")
# displacy.serve(doc, style="dep", options=options)
# actor_classification()
# collect_non_elections_data()
# evaluate()

# all_summary = summarizer.summarize('Summarizer/Milestone3 - Copy.csv', True)
# print('------ Summarized Results ---------')
#
# final_summary1 = ' '.join(all_summary[0])
# final_summary1 = final_summary1.replace('\n', ' ')
# print(final_summary1)