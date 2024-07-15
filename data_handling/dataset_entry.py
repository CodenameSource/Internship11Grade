class Entry:
    def __init__(self, image, ocr, entities):
        self.image = image
        self.ocr = ocr
        self.entities = {"hero": [], "villain": [], "victim": [], "other": []}
        for e_class in entities:
            if e_class in self.entities:
                for e in entities[e_class]:
                    self.entities[e_class].append(e)

