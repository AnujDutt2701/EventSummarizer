from allennlp.predictors.predictor import Predictor
import spacy as spacy
import re
predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")

# Extracting relationships for:
#
# In June, an army soldier Aurangzeb of 44 Rashtriya Rifles posted in south Kashmir\x92s Shopian district was
# abducted by militants and his bullet-riddled body was found 10 kilometers away from the place of kidnapping.
# Soon after police came to know about abduction of the soldier, a joint police and army team was send to village
# and search operation was launched in the area.Officials said that Bhat was at his home at Qazipora when some
# unidentified gunmen abducted him from his house.


def get_relation(summary,root_verb):
    results1 = predictor.predict(summary)
    arg0 = '-'
    arg1 = '-'
    for word, verb in zip(results1["words"], results1["verbs"]):
        if (verb["verb"] == root_verb):
            text = verb["description"]
            args = re.findall('\[(.*?)\]', text)
            print(args)
            for item in args:
                if 'ARG0' in item:
                    arg0 = item.split(':')[1]
                elif 'ARG1' in item:
                    arg1 = item.split(':')[1]
            return arg0, arg1
    return arg0, arg1
