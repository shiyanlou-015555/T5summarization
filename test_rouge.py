from dataclasses import dataclass
import rouge
rouge_score = rouge.Rouge()
# scores = rouge_score.get_scores(hyp_path, ref_path, avg=True)
import json
data = json.load(open("/data1/ach/project/T5summarization/tagged/test.json"))
pass