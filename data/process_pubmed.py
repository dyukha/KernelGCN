from typing import Dict, List

with open("Pubmed-Diabetes.DIRECTED.cites.tab") as fin:
    fin.readline()
    fin.readline()

    with open("pubmed.cites", "w") as fout:
        for line in fin:
            data = line.rstrip().split()
            u, v = [paper.split(':')[1] for paper in [data[1], data[3]]]
            print(u, v, file=fout)

class Example:
    def __init__(self, id: int, features: Dict[str, str], clazz: int):
        self.id = id
        self.features = features
        self.clazz = clazz

examples: List[Example] = []
all_features = set()
with open("Pubmed-Diabetes.NODE.paper.tab") as fin:
    fin.readline()
    fin.readline()

    for line in fin:
        v, *data = line.rstrip().split()
        features: Dict[str, str] = dict([x.split("=") for x in data])
        clazz = int(features["label"])
        del features["label"]
        del features["summary"]
        all_features.update(features.keys())
        examples.append(Example(int(v), features, clazz))

print(len(all_features))
all_features = list(all_features)
print(all_features[:100])

with open("pubmed.content", "w") as fout:
    for e in examples:
        print(e.id, end=" ", file=fout)
        for f in all_features:
            print(e.features.get(f, "0"), end=" ", file=fout)
        print(e.clazz, file=fout)