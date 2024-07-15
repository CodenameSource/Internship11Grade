from PIL import Image
from textblob import TextBlob
from NewsSentiment import TargetSentimentClassifier

text = """
The meme is a playful and humorous representation of the Wuhan Coronavirus (COVID-19) pandemic. It uses the "Online, the line is used to punctuate a joke" template, where one person admits to something humorous, absurd, or offensive. In this case, the "Wuhan kid" is admitting that he is responsible for infecting people with wars. This is a satirical reference to the idea that the Wuhan Coronavirus originated from a lab in Wuhan, China, and the possible connection between the virus and global conflicts. The "teacher" in the meme represents someone who is questioning the "Wuhan kid," while the "China virus meme #3" indicates that this is the third instance of such a meme using the same format."""

blob = TextBlob(text)

print(blob)
tsc = TargetSentimentClassifier()

data = [
    ("I like ", "Peter", " but I don't like Robert."),
    ("", "Mark Meadows", "'s coverup of Trump’s coup attempt is falling apart."),
    ('The meme is a playful and humorous representation of the Wuhan Coronavirus (COVID-19) pandemic. It uses the "Online, the line is used to punctuate a joke" template, where one person admits to something humorous, absurd, or offensive. In this case, the ', "Wuhan kid", ' is admitting that he is responsible for infecting people with wars. This is a satirical reference to the idea that the Wuhan Coronavirus originated from a lab in Wuhan, China, and the possible connection between the virus and global conflicts. The "teacher" in the meme represents someone who is questioning the "Wuhan kid," while the "China virus meme #3" indicates that this is the third instance of such a meme using the same format.'),
    ('The meme is a playful and humorous representation of the Wuhan Coronavirus (COVID-19) pandemic. It uses the "Online, the line is used to punctuate a joke" template, where one person admits to something humorous, absurd, or offensive. In this case, the "Wuhan kid" is admitting that he is responsible for infecting people with wars. This is a satirical reference to the idea that the Wuhan Coronavirus originated from a lab in Wuhan, China, and the possible connection between the virus and global conflicts. The ', "teacher", 'in the meme represents someone who is questioning the "Wuhan kid," while the "China virus meme #3" indicates that this is the third instance of such a meme using the same format.'),
]

sentiments = tsc.infer(targets=data)

for i, result in enumerate(sentiments):
    print("Sentiment: ", i, result[0])
