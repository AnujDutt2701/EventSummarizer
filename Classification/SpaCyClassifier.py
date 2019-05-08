import random
import spacy
from pathlib import Path
from DataPreparation import PreProcessData as ppd
from spacy.util import minibatch, compounding
import pandas as pd


def classifier(model=None, output_dir=None, n_iter=20, n_texts=2000):
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    nlp = spacy.load(model)
    print("The model loaded is: '%s'" % model)

    # if model is not None:
    #     nlp = spacy.load(model)  # load existing spaCy model
    #     print("Loaded model '%s'" % model)
    # else:
    #     nlp = spacy.blank("en")  # create blank Language class
    #     print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn", "exclusive_classes": True}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    # add label to text classifier
    textcat.add_label("RiotsProtests")
    textcat.add_label("ViolenceAgainstCivilians")



    print("Loading training data...")
    (train_notes, train_event_type), _ = load_data()
    # train_notes = train_notes[: n_texts]
    # train_event_type = train_event_type[: n_texts]
    train_data = list(zip(train_notes, [{"cats": cats} for cats in train_event_type]))

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        # if init_tok2vec is not None:
        #     with init_tok2vec.open("rb") as file_:
        #         textcat.model.tok2vec.from_bytes(file_.read())
        print("Training the model...")
        # print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        for i in range(n_iter):
            print("The current iteration is " + str(i))
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            # with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
            #     scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            # print(
            #     "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
            #         losses["textcat"],
            #         scores["textcat_p"],
            #         scores["textcat_r"],
            #         scores["textcat_f"],
            #     )
            # )

    # test the trained model
    # test_text = "On February 9, a Trinamool Congress (TMC) legislator was shot dead by unidentified armed men at Phulbari city (Jalpaiguri, West Bengal)."
    # doc = nlp(test_text)
    # print(test_text, doc.cats)

    if output_dir is not None:
        with nlp.use_params(optimizer.averages):
            nlp.to_disk(output_dir)
        print("Saved model to", output_dir)


def load_data(limit=0, split=0.8):
    df = ppd.read_data('../DataPreparation/2018-04-19-2019-04-17.csv',60000, False)
    train_data = [tuple(x) for x in df.values]
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    notes, labels = zip(*train_data)
    event_types = [{"RiotsProtests": not bool(y), "ViolenceAgainstCivilians":  bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (notes[:split], event_types[:split]), (notes[split:], event_types[split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


def classify(text, dir, train):
    output_dir = dir
    if train:
        classifier('en',output_dir)
    print("Loading classifier from ", output_dir)
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(text)
    print(doc2.cats)
    is_protest = 'Protest/Riot' if doc2.cats['RiotsProtests'] > doc2.cats['ViolenceAgainstCivilians'] else 'Violence Against Civilians'
    return is_protest


def predict(text, model_dir):
    nlp2 = spacy.load(model_dir)
    doc2 = nlp2(text)
    return doc2.cats


def evaluate_model(model_dir, df):
    nlp = spacy.load(model_dir)
    eval_data = [tuple(x) for x in df.values]
    notes, labels = zip(*eval_data)
    event_types = [{"RiotsProtests": not bool(y), "ViolenceAgainstCivilians": bool(y)} for y in labels]
    cats = [{"RiotsProtests": not bool(y), "ViolenceAgainstCivilians":  bool(y)} for y in event_types]
    evaluate(nlp.tokenizer, nlp, notes, cats)


def classify_texts(text_array, dir, target_array):
    output_dir = dir
    # classifier('en','spacy_classifier')
    print("Loading classifier from output directory: ", output_dir)
    nlp2 = spacy.load(output_dir)
    correct = 0
    wrong = 0

    for index, text in enumerate(text_array):
        doc2 = nlp2(text)
        print(text, doc2.cats)
        print('target is: ' + str(target_array[index]))
        is_protest = doc2.cats['RiotsProtests'] > doc2.cats['ViolenceAgainstCivilians']
        predicted_target = 0 if is_protest else 1
        actual_target = target_array[index]
        actual_target = 0 if actual_target['RiotsProtests'] else 1
        if actual_target == predicted_target:
            correct = correct + 1
        else:
            wrong = wrong + 1
    accuracy = correct/ (correct + wrong)

    print('The accuracy is: ' + str(accuracy))

# classify("test","event_classifier")

# To check the accuracy of the classifier
# _, testdata = load_data()
# classify_texts(testdata[0],'spacy_classifier', testdata[1])

# if __name__ == "__main__":
#     # train_data, _ = thinc.extra.datasets.imdb()
#     output_dir = 'spacy_classifier'
#     #classifier('en','spacy_classifier')
#     print("Loading from", output_dir)
#     nlp2 = spacy.load(output_dir)
#     test_text = ['''India has lodged a protest with Pakistan over four separate incidents of
# alleged harassment of Indian High Commission officials in Islamabad and sought
# an immediate investigation into them, official sources said Thursday.
# On March 18, India issued a note verbale  a diplomatic communication  to the
# Pakistan Foreign Ministry, giving a detailed account of the incidents,
# including aggressive tailing of the Indian naval adviser by two Pakistani
# personnel, they said.
# India issued a similar note
# verbale to the Pakistan Foreign Ministry on March 13, lodging a strong protest
# over several incidents of alleged harassment of Indian High Commission
# officials between March 8 and 11.
# In a separate
# case, the note said another official of the mission received hoax calls while
# another staff of the mission was subjected to intimidatory behaviour by
# Pakistani personnel on March 14, the sources said.
# In the note, India asked Pakistan to carry out immediate investigation into
# the cases of harassment of Indian officials.''']
#     # nlp2 = spacy.load('spacy_classifier')
#
#     for text in test_text:
#         doc2 = nlp2(text)
#         print(text, doc2.cats)
# classify("On 11 April, one person was killed and three others were injured in a mob attack over cutting flesh of a dead bull in Gumla district (Jharkhand)",
#          "spacy_classifier_new", True)


# load_data()