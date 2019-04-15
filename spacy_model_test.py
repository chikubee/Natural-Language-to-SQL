from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

#@author: ANKITA MAKKER
# training data
TRAIN_DATA = [
("I'm taking algebra this semester.", {"entities": [(11, 18, "COURSE")]}),
("He had always hated biology and chemistry.", {"entities": [(20, 27, "COURSE"), (32, 41,"COURSE")]}),
("He decided to take two philosophy classes his senior year.", {"entities": [(20, 30, "COURSE")]}),
("She studied psychology in college.", {"entities": [(12, 22, "COURSE")]}),
("They had English together two years in a row.", {"entities": [(9, 16, "COURSE")]}),
("He is hoping to take French next year.", {"entities": [(21, 26, "COURSE")]}),
("She was really enjoying geometry.", {"entities": [(24, 31, "COURSE")]}),
("She challenged herself by taking Physics 301.", {"entities": [(32, 42, "COURSE")]}),
("His first year in college he took Philosophy of Language, Math 101, and Educational Psychology.", {"entities": [(34, 55, "COURSE"), (58, 65, "COURSE"), (72, 93, "COURSE") ]}),
("Mike teaches Physics", {"entities": [(13, 20, "COURSE")]}),
("Jon failed in Mathematics last semester", {"entities": [(13, 24, "COURSE")]}),
("Mary scored 89 in Commerce", {"entities": [(18, 26, "COURSE")]}),

    # ("My zip code is 482005.", {"entities": [(15, 21, "ZIP")]}),
    # ("Berlin has a code of 123456.", {"entities": [(21, 27, "ZIP")]}),
    # ("The person living in my neighbourhood does not belong to 908768.", {"entities": [(59, 65, "ZIP")]}),
    # ("My zip code is 483456.", {"entities": [(15, 21, "ZIP")]}),
    # ("My zip code is 879987.", {"entities": [(15, 21, "ZIP")]}),
    # ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    # ("Rishti is a student of SHSSS.", {"entities": [(0, 6, "PERSON")]}),
    # ("Joey is a character in FRIENDS.", {"entities": [(0, 4, "PERSON")]}),
    # ("I know risht", {"entities": [(7, 12, "PERSON")]}),
    # ("Rama makes rules everytime.", {"entities": [(0, 4, "PERSON")]}),
    # ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, output_dir=None, n_iter=100):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly – but only if we're
        # training a new model
        if model is None:
            nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    test = "list of students enrolled for Physics and Mathematics"
    doc = nlp(test)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
    # for text, _ in TRAIN_DATA:
    #     doc = nlp(text)
    #     print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
    #     print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    
    nlp.to_disk("retrained_en_model")
    nlp2 = spacy.load("retrained_en_model")
    for text, _ in TRAIN_DATA:
        doc = nlp2(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        for text, _ in TRAIN_DATA:
            doc = nlp2(text)
            print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
            print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])


if __name__ == "__main__":
    plac.call(main)

    # Expected output:
    # Entities [('Shaka Khan', 'PERSON')]
    # Tokens [('Who', '', 2), ('is', '', 2), ('Shaka', 'PERSON', 3),
    # ('Khan', 'PERSON', 1), ('?', '', 2)]
    # Entities [('London', 'LOC'), ('Berlin', 'LOC')]
    # Tokens [('I', '', 2), ('like', '', 2), ('London', 'LOC', 3),
    # ('and', '', 2), ('Berlin', 'LOC', 3), ('.', '', 2)]