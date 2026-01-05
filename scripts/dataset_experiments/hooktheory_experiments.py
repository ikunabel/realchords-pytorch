import json
import gzip

with gzip.open("data/hooktheory/Hooktheory.json.gz", "r") as f:
    dataset = json.load(f)

song = dataset["qveoYyGGodn"]

#Metadata
# print(song['tags'])
# print(song['hooktheory'])
# print(song['youtube'])

# print("\n Aligment: \n")
# print(song['alignment'])

# print("\n Annotations: \n")
# print(song['annotations'])

print(dataset.items())
