import spacy
from spacy.training.example import Example

nlp = spacy.load("en_core_web_sm")
ner = nlp.get_pipe("ner")
ner.add_label("POKEMON")

TRAIN_DATA = [
    (
        "In the heart of the forest, Pikachu darted between the trees, its cheeks sparking with electricity. Charizad soared overhead, casting a shadow on the ground below. Mewto watched silently from a distance, while Bulbasaur napped under a large leaf.",
        {"entities": [(32, 39, "POKEMON"), (98, 106, "POKEMON"), (146, 151, "POKEMON"), (181, 190, "POKEMON")]}
    ),
    (
        "Charizad and Pikachu teamed up to face Mewto in an epic battle. Bulbasaur cheered from the sidelines, its vines waving excitedly.",
        {"entities": [(0, 8, "POKEMON"), (13, 20, "POKEMON"), (39, 44, "POKEMON"), (70, 79, "POKEMON")]}
    ),
    (
        "Bulbasaur, Pikachu, and Charizad explored the ancient ruins, unaware that Mewto was observing their every move.",
        {"entities": [(0, 9, "POKEMON"), (11, 18, "POKEMON"), (24, 32, "POKEMON"), (68, 73, "POKEMON")]}
    ),
    (
        "Mewto challenged Charizad to a duel at sunrise. Pikachu and Bulbasaur watched with anticipation as the two powerful Pokémon clashed.",
        {"entities": [(0, 5, "POKEMON"), (17, 25, "POKEMON"), (39, 46, "POKEMON"), (51, 60, "POKEMON")]}
    ),
    (
        "Pikachu ran circles around Bulbasaur, while Charizad tried to keep up. Mewto simply levitated above them, amused by their antics.",
        {"entities": [(0, 7, "POKEMON"), (25, 34, "POKEMON"), (42, 50, "POKEMON"), (72, 77, "POKEMON")]}
    ),
    (
        "The four friends—Charizad, Pikachu, Mewto, and Bulbasaur—set out on a journey across the land, facing many challenges together.",
        {"entities": [(19, 27, "POKEMON"), (29, 36, "POKEMON"), (38, 43, "POKEMON"), (49, 58, "POKEMON")]}
    ),
    (
        "Charizad unleashed a fiery blast, but Pikachu dodged and countered with a thunderbolt. Mewto shielded Bulbasaur from the shockwave.",
        {"entities": [(0, 8, "POKEMON"), (44, 51, "POKEMON"), (81, 86, "POKEMON"), (96, 105, "POKEMON")]}
    ),
    (
        "Bulbasaur and Mewto worked together to solve the puzzle, while Pikachu and Charizad scouted ahead for danger.",
        {"entities": [(0, 9, "POKEMON"), (14, 19, "POKEMON"), (56, 63, "POKEMON"), (68, 76, "POKEMON")]}
    ),
    (
        "Pikachu, Bulbasaur, Charizad, and Mewto gathered around the campfire, sharing stories of their adventures.",
        {"entities": [(0, 7, "POKEMON"), (9, 18, "POKEMON"), (20, 28, "POKEMON"), (34, 39, "POKEMON")]}
    ),
    (
        "As the sun set, Mewto meditated quietly, while Charizad practiced its flying. Pikachu and Bulbasaur played nearby.",
        {"entities": [(16, 21, "POKEMON"), (44, 52, "POKEMON"), (66, 73, "POKEMON"), (78, 87, "POKEMON")]}
    ),
]

optimizer = nlp.resume_training()
for itn in range(10):
    for text, annotations in TRAIN_DATA:
        example = Example.from_dict(nlp.make_doc(text), annotations)
        nlp.update([example], drop=0.5, sgd=optimizer)

nlp.to_disk("custom_model")
nlp = spacy.load("custom_model")
# ...existing code...

# Test the custom model
test_text = "HQ has detected unusual Bulbasaur activity in the area. Field sensors logged anomalous behavior that suggests an imminent threat. Remember there are Pikachu and Charizard nearby — take care not to draw them into combat. You are to neutralize the bulbasaurs immediately. Report status once the target is down. Confirm mission status and any collateral damages."
doc = nlp(test_text)
print("Entities:", [(ent.text, ent.label_) for ent in doc.ents])
#
