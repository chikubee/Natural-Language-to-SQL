import spacy
nlp = spacy.load('en')
doc = nlp(u"For how many hours does Mike teach Physics?")
for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
          [child for child in token.children])