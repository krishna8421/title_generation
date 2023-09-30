import json
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

if nltk.data.find('tokenizers/punkt'):
    print("NLTK resources found")
else:
    print("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


test_data_json = json.load(open("data/raw/test.json"))
train_data_json = json.load(open("data/raw/train.json"))
valid_data_json = json.load(open("data/raw/valid.json"))

if os.path.exists("data/processed"):
    os.system("rm -rf data/processed")

output_dir = "data/processed"
os.makedirs(output_dir, exist_ok=True)


def process_text(text):
    text = text.lower()

    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\d+', '', text)

    words = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    cleaned_text = ' '.join(words)

    return cleaned_text



def process_dataset(data_json, output_file):
    print("Processing data...")
    data = []
    for _, conversation_data in enumerate(data_json):
        title = conversation_data["title"].lower().replace("'", "")
        content = process_text(conversation_data["content"])

        data.append(title + " $@$ " + content)

    filename = os.path.join(output_dir, output_file)
    with open(filename, "w") as file:
        file.write("\n".join(data))

    print(f"Data saved to {filename}")

# Process the data
process_dataset(test_data_json, "test.txt")
process_dataset(train_data_json, "train.txt")
process_dataset(valid_data_json, "valid.txt")


