import json
from json import dumps, loads
from functools import reduce

a = {"objekt": "jablko, macka", "pole": []}

with open("chatgpt_questions_results\\results_20241218_195736.json", 'r', encoding='utf-8') as json_file:
    out = json.loads(json_file.read())

#zoznam = [float(x["numerical_response"]) for x in out["responses"] if x["numerical_response"] != '2.5']
zoznam = [float(x["numerical_response"]) for x in out["responses"]]

zoznam = list(set(zoznam))

zoznam2 = sorted(zoznam, reverse=True, key=lambda x: 1/x)

print(zoznam2)

product = reduce(lambda x,y: x*y, zoznam, 1)
print(product)
print(zoznam)


p1 = [1, 2, 3]
p2 = ["a", "b", 'c']

p3 = [str(x) + y for x,y in zip(p1, p2)]

p4 = p1*3
p5 = p2 + p3

print(p3)
print(p4)
print(p5)


