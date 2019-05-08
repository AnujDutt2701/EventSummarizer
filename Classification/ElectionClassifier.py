import pandas
import Article_Parser as scrapper
import random
import spacy
from pathlib import Path
from DataPreparation import PreProcessData as ppd
from spacy.util import minibatch, compounding
import pandas as pd


def collect_non_elections_data():
    raw_data = pandas.read_csv("../Election_Demo.csv")
    print(raw_data["url"])
    df = pandas.DataFrame(columns=['X', 'Y'])
    articles = []
    target = []
    for url in raw_data["url"]:
        # articles.append(scrapper.get_article(url))
        # target.append(0)
        raw_content = scrapper.get_article(url)
        if raw_content:
            content = raw_content["content"]
            df = df.append({'X': content, 'Y': 0}, ignore_index=True)
    print(df)
    df.to_csv("Election_Demo_articles.csv")
    print(df)


# collect_non_elections_data()


def classifier(model=None, output_dir=None, n_iter=20, n_texts=2000):
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    nlp = spacy.load(model)
    print("The model loaded is: '%s'" % model)


    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat", config={"architecture": "simple_cnn", "exclusive_classes": True}
        )
        nlp.add_pipe(textcat, last=True)
    else:
        textcat = nlp.get_pipe("textcat")

    # add label to text classifier
    # Below are the sub categories for Violence against civilian
    textcat.add_label("Yes")
    textcat.add_label("No")

    print("Loading training data...")
    (article, article_type), _ = load_data()
    train_data = list(zip(article, [{"cats": sub_event_type} for sub_event_type in article_type]))

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
                # print(annotations)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            # with textcat.model.use_params(optimizer.averages):
            #evaluate on the dev data split off in load_data()
                #scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
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
    df = ppd.article_data()
    train_data = [tuple(x) for x in df.values]
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    notes, labels = zip(*train_data)
    event_types = [{"Yes": bool(y), "No": not bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (notes[:split], event_types[:split]), (notes[split:], event_types[split:])

def classify(text, dir):
    output_dir = dir
    #classifier('en', output_dir)
    print("Loading classifier from ", output_dir)
    nlp2 = spacy.load(output_dir)
    doc2 = nlp2(text)
    print(doc2.cats)
    # sub_event = max(doc2.cats.items(), key=operator.itemgetter(1))[0]
    # print(sub_event)
    # return sub_event


# classify('''On April 24, supporters of a CPI candidate were assaulted by BJP workers a few distances away from the polling station in Yambem village under Andro assembly constituency area (Imphal East, Manipur), during repolling of the Parliamentary elections.''',
#          "article_classifier")