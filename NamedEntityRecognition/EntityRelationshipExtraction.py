from DataPreparation import PreProcessData as ppd
from pathlib import Path
from spacy import displacy
import NamedEntityRecognition.NETagging as ner
import spacy
from spacy.pipeline import EntityRuler
import NamedEntityRecognition.allennlpner as allen


# TODO: Move it to locations code
# locations_tags = ['NORP', 'GPE', 'LOC']
# all_entities = [(ent.text, ent.label_) for ent in doc.ents]
# locations = [loc for loc,tag in all_entities if locations_tags.__contains__(tag)]
# print(all_entities)
#displacy.serve(doc, style="dep")


def print_dependency_pattern(doc):
    dependency_pattern = '{left}<---{word}[{w_type},{pos_tag}]--->{right}\n--------'
    for token in doc:
        print(dependency_pattern.format(word=token.orth_,
                                      w_type=token.dep_,
                                        pos_tag = token.pos_,
                                      left=[t.orth_
                                                for t
                                                in token.lefts],
                                      right=[t.orth_
                                                 for t
                                                 in token.rights]))


def get_inner_related_entities(subject):
    # prep_right = [t for t in subject.rights if t.dep_ == 'prep']
    # right_text = ''
    # if len(prep_right) > 0:
    #     right_text =right_text + " " +prep_right[0].orth_ + " "
    #     prep_right_obj = [t for t in prep_right[0].rights if 'obj' in t.dep_]
    #     if len(prep_right_obj) > 0:
    #         right_text = right_text + " " + prep_right_obj[0].orth_+ " "

    noun_left = [t for t in subject.lefts if t.dep_ == 'compound']
    noun_right = [t for t in subject.rights if t.dep_ == 'compound']
    prop_noun = ''
    if len(noun_left) > 0:
        prop_noun = ' '.join([t.orth_ for t in noun_left])
    if len(noun_right) > 0:
        prop_noun = ' '.join([t.orth_ for t in noun_right])
    # if right_text:
    #     return subject.orth_ + " " + right_text
    return prop_noun + " " + subject.orth_


def get_related_entities(root):
    if 'subj' in root.dep_ or root.pos_ == 'NOUN' or root.pos_ == 'PROPN':
        return get_inner_related_entities(root)
    if len([t for t in root.lefts]) == 0 and len([t for t in root.rights]) == 0:
        return ''
    left_child = ''
    for t in root.lefts:
        if left_child:
            break
        left_child = get_related_entities(t)

    right_child = ''
    for t in root.rights:
        if right_child:
            break
        right_child = get_related_entities(t)

    if left_child != '' and right_child != '':
        return left_child + ", " + right_child
    elif left_child:
        return left_child
    elif right_child:
        return right_child
    else:
        return ''


def get_relationship(document, printEntites, serve):
    related_entities = '-, -'
    relation = ''
    spacy_nlp = ner.get_ner()
    docs = spacy_nlp(document)
    docs_first = spacy_nlp(document.split(".")[0])
    print('------ Identified Entities are: ---------')
    if printEntites:
        for element in docs.ents:
            print('Type: %s, Value: %s' % (element.label_, element))

    for token in docs:
        if token.dep_ == 'ROOT' and (token.n_lefts != 0 and token.n_rights != 0) and token.text == token.head.text:
            relation = token.text
            arg0, arg1 = allen.get_relation(document, relation)

            related_entities = arg0 + ", " + arg1
            # related_entities = get_related_entities(token)
            break
    print(relation + " relation is in between " + related_entities)
    dep_html = ''
    ent_html = ''

    if serve:
        options = {"bg": "#ffffff",
                   "color": "#212121", "font": "Roboto"}
        dep_html = displacy.render(docs_first, style="dep", options=options)
        print(dep_html)
        print("--------------")
        options = {"bg": "#ffffff",
                   "color": "#212121", "font": "Roboto"}
        ent_html = displacy.render(docs, style="ent", options=options)
        print(ent_html)

    return relation, related_entities, dep_html, ent_html

def get_entites(document):
    spacy_nlp = ner.get_ner()
    doc = spacy_nlp(document)
    entities = []
    for element in doc.ents:
        temp = [element.label_, str(element)]
        entities.append(temp)
        print('Type: %s, Value: %s' % (element.label_, element))
    print(entities)
    return entities


