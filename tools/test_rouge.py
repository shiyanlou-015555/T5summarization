from dataclasses import dataclass
from math import hypot
# import rouge
# rouge_score = rouge.Rouge()
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)
hypo = "A French prosecutor says he is not aware of any video footage from on board the plane. German daily Bild and Paris Match claim to have found a cell phone video of the crash. A French Gendarmerie spokesman calls the reports \"completely wrong\" and \"unwarranted\""
target = "Marseille prosecutor says \"so far no videos were used in the crash investigation\" despite media reports . Journalists at Bild and Paris Match are \"very confident\" the video clip is real, an editor says . Andreas Lubitz had informed his Lufthansa training school of an episode of severe depression, airline says ."
scores = scorer.score(hypo,target)
print(scores)
# import json
# data = json.load(open("/data1/ach/project/T5summarization/tagged/test.json"))
# pass