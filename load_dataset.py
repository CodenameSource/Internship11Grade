import pandas as pd
import spacy
import re
import imagehash

dataset_folder = "data/HVVMemes"
annotations_folder = "data/HVVMemes/annotations"

def entities_in_text(entities, text):
    count = 0
    e_text = []
    for e in entities:
        if e in text:
            count += 1
            e_text.append(e)

    if len(entities) == 0:
        return 1, []
    return count / len(entities), e_text

def check_entries(df):
    avg_ent_count = 0
    e_in_text = []
    e_not_in_text = []
    for idx, row in train_df.iterrows():
        text = row["OCR"]
        ent = []
        ent.extend(row["hero"] + row["villain"] + row["victim"] + row["other"])

        i, e = entities_in_text(ent, text.lower())
        avg_ent_count += i
        e_in_text.append(e)
        e_not_in_text.append([d for d in ent if d not in e])

    return avg_ent_count / len(df)

if __name__ == "__main__":
    train_df = pd.read_json(f"{annotations_folder}/train.jsonl", lines=True)

    print(f"Average entities in text: {check_entries(train_df)}")