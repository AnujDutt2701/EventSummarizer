from DataPreparation import PreProcessData as ppd
from pathlib import Path
import spacy
from spacy.pipeline import EntityRuler

spacy_nlp = ''
def get_ner():
    global spacy_nlp
    if spacy_nlp:
        return spacy_nlp
    spacy_nlp= spacy.load('en')
    ruler = EntityRuler(spacy_nlp)
    ruler.from_disk("NamedEntityRecognition/entities.jsonl")
    spacy_nlp.add_pipe(ruler, before="ner")
    output_dir = Path('CustomNER')
    if not output_dir.exists():
        output_dir.mkdir()
        spacy_nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    # TODO: Check how to save and load from disk
    # print("Loading from", output_dir)
    # spacy_nlp = spacy.load("en", path=output_dir)
    # doc = spacy_nlp(u"To calm down the entire Panna situation, District Educational Officer of Ambedkar Nagar had to intervene. The DEO listened to the grievances of the students and assured that their concerns would be forwarded to higher authorities. Only after DEO’s intervention, the students called-off their protest.")
    # print([(ent.text, ent.label_) for ent in doc.ents])
    return spacy_nlp

# document = '''North Garo Hills BJP Kisan Morcha deputy police North India chairman of Ambedkar Nagar was shot Krishna dead at his Paharpura village under the Haspura police station area on Friday morning police. Madan, along with his two acquaintances, was on morning walk when two unidentified criminals shot him in his temple at a very close range near Jalpura Mor. He died on the spot. Sources said the criminals shot him came out for walk with his two acquaintances at 5:30 in the morning and fled. The murder created tension in the area and shops in Haspura Bazar remained closed till 2pm. Daudnagar SDPO Rajkumar Tiwari said an FIR has been lodged and two persons have been interrogated. Sources said Madan was RJD’s Haspura block president before joining the BJP before the 2015 assembly elections. He was also running a PDS shop at his native Paharpura village. Madan’s supporters gathered in large numbers and demanded proper compensation to his family. BJP MLA Manoj Kumar Sharma and SP Satyprakash rushed to the spot. Only when Sharma assured people to get justice done that they allowed the police to send the body for postmortem.'''
# document = '''Apple is opening its first big office in San Francisco.'''
# document = spacy_nlp(document)
#
# print(document.ents)
# for element in document.ents:
#     print('Type: %s, Value: %s' % (element.label_, element))

# print([(X, X.ent_iob_, X.ent_type_) for X in document])
