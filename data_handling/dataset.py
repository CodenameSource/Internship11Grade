import json
import os

from data_handling.dataset_entry import Entry


class Dataset:
    def __init__(self, root_folder, train_annotations, val_annotations, test_annotations):
        self.train = []
        self.val = []
        self.test = []

        self.root_folder = root_folder
        self.train_annotations = train_annotations
        self.val_annotations = val_annotations
        self.test_annotations = test_annotations

        self._load_annotations()

    def _load_annotations(self):
        with open(self.train_annotations, "r") as f:
            train = json.load(f)
            for entry in train:
                image = os.path.join(self.root_folder, "images", entry["image"])
                ocr = entry["ocr"]
                entities = entry["entities"]
                self.train.append(Entry(image, ocr, entities))
        with open(self.val_annotations, "r") as f:
            val = json.load(f)
            for entry in val:
                image = os.path.join(self.root_folder, "images", entry["image"])
                ocr = entry["ocr"]
                entities = entry["entities"]
                self.val.append(Entry(image, ocr, entities))
        with open(self.test_annotations, "r") as f:
            test = json.load(f)
            for entry in test:
                image = os.path.join(self.root_folder, "images", entry["image"])
                ocr = entry["ocr"]
                entities = entry["entities"]
                self.test.append(Entry(image, ocr, entities))