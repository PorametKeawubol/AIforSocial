import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")
with open("The_Gift_of_the_Magi_O_Henry.txt", "r", encoding="utf-8") as f:
    text = f.read()

doc = nlp(text)

# Named Entity Visualization
displacy.serve(doc, style="ent")

# หรือถ้าอยากดู Dependency Tree ให้ใช้:
# displacy.serve(doc, style="dep", options={"compact": True, "bg": "#fafafa"})
