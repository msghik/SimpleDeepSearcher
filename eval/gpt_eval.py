from openai import OpenAI
import sys
import os
from datasets import load_dataset
import http.client
import json
import re
from tqdm import tqdm
import multiprocessing
from time import sleep
import requests
import json
from collections import defaultdict
import random
import json
import requests
import time
# from bs4 import BeautifulSoup
# import wikipediaapi
from urllib.parse import unquote
from urllib.parse import urlparse


os.environ["OPENAI_API_KEY"]  = "YOUR_API_KEY"
os.environ["OPENAI_API_BASE"] = "YOUR_API_BASE"

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)


def generate(messages, model_name):
    retry_cnt = 0
    while True: # 重试机制
        try:
            if retry_cnt:
                print(f"Retry: {retry_cnt}")
            response = client.chat.completions.create(
                **{
                    "model": model_name,
                    "messages": messages,
                    "max_tokens": 1024,
                    "temperature": 0,
                }
            )
            response = response.choices[0].message.content
            return response
        except Exception as e:
            retry_cnt += 1
            print(f"Error: {e}")
            time.sleep(0.5)
def process_one_sample(obj):

    prompt = '''Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. Respond with True if the prediction is correct and False otherwise.
Golden Answer may have multiple options, and matching any one of them is considered correct.

Question: {question}
Golden Answer: {reference}
Predicted Answer: {prediction}
    '''

    question = obj["item"]["Question"]
    reference_ans_ori = obj["item"]["answer"]

    reference_ans_ori = [a if isinstance(a, str) else str(a) for a in reference_ans_ori]
    if isinstance(reference_ans_ori, str):
        reference_ans = reference_ans_ori
    elif isinstance(reference_ans_ori, list):
        reference_ans = "; ".join(reference_ans_ori)
        # print(reference_ans)
    else:
        raise ValueError(
            f"Unsupported type for reference_ans_ori: {type(reference_ans_ori)}"
        )

    if reference_ans ==False:
        reference_ans="no"
    if reference_ans ==True:
        reference_ans="yes"
    solution = obj["output"]

    pattern = r'\\boxed\{(.*)\}'
    matches = re.findall(pattern, solution)
    # proposed_ans = matches[-1]
    if matches:
        proposed_ans = matches[-1]
    else:
        proposed_ans = "No answer"


    gpt4o_input = prompt.format(question=question , reference=reference_ans, prediction=proposed_ans)
    flag_final_ans = True

    if flag_final_ans:
        messages = [{'role': 'user', 'content': gpt4o_input}]
        model_output = generate(messages, 'gpt-4o-mini')
        obj["gpt4o_output"] = model_output
    else:
        obj["gpt4o_output"] = "Fuck! No boxed"

    obj_new={
        "question":question,
        "reference_ans":reference_ans ,
        "predicted_ans": proposed_ans ,
        "gpt4o_output":obj["gpt4o_output"] ,
        "source": obj["item"]["source"]
        # "output":solution,
    }

    return obj_new


def cal_metrics(results):
    metrics = {
        "is_correct": 0,
        "is_incorrect": 0,
        "invalid_judge": 0,
        "num": 0
    }

    error_cnt = 0
    source_metrics = {}

    for sample in results:
        source = sample["source"]
        if source not in source_metrics:
            source_metrics[source] = {
                "is_correct": 0,
                "is_incorrect": 0,
                "invalid_judge": 0,
                "num": 0
            }
        
        # 统计数目
        metrics["num"] += 1
        source_metrics[source]["num"] += 1

        # 判断是否正确
        if sample["gpt4o_output"] == "True":
            metrics["is_correct"] += 1
            source_metrics[source]["is_correct"] += 1
        elif sample["gpt4o_output"] == "False":
            metrics["is_incorrect"] += 1
            source_metrics[source]["is_incorrect"] += 1
        else:
            metrics["invalid_judge"] += 1
            source_metrics[source]["invalid_judge"] += 1
            error_cnt += 1
    

    print("Total:", metrics["num"])
    print(f"error_cnt: {error_cnt}")

    # 计算汇总指标
    for key in metrics:
        if key == "num":
            continue
        metrics[key] = metrics[key] / metrics["num"] if metrics["num"] > 0 else 0
    
    # 计算各个source
    for src in source_metrics:
        for key in source_metrics[src]:
            if key == "num":
                continue
            source_metrics[src][key] = source_metrics[src][key] / source_metrics[src]["num"] if source_metrics[src]["num"] > 0 else 0
    

    final_metrics = {'overall': metrics, 'per_source': source_metrics}

    return final_metrics



if __name__ == '__main__':
    input_files=[
        ".../turn_(last_turn).json" # such as "output/bamboogle/turn_2.json"
    ]   

    for input_file in input_files:
        print(f"Begin:{input_file}")

        output_file = input_file.replace(".json", "_judge.json")
        chunk_size=200
        with open(input_file, "r") as fin:
            all_demons = json.load(fin)  # 加载整个 JSON 文件，返回一个列表
        
        for item in all_demons:
            if 'source' not in item["item"]:
                item["item"]["source"] = 'unknown'
        # all_demons =all_demons [:1]
        print(f"Processed data has been written to {output_file}")

        print("All Data Num:",len(all_demons))
        chunk_num = len(all_demons) // chunk_size
        if len(all_demons) % chunk_size != 0:
            chunk_num += 1


        all_results = []
        for chunk_i in range(chunk_num):
            print("Epoch:" ,chunk_i ,"/",chunk_num)
            all_demons_subset = all_demons[chunk_i*chunk_size : (chunk_i+1)*chunk_size]
            print(len(all_demons_subset))
            with multiprocessing.Pool(processes=200) as pool:
                results = list(tqdm(pool.imap(process_one_sample, all_demons_subset), total=len(all_demons_subset)))

            all_results.extend(results)
        
        # 保存为json

        final_metrics = cal_metrics(all_results)

        with open(output_file, 'w') as fout:
            json.dump(all_results, fout, ensure_ascii=False, indent=4)

        with open(output_file.replace(".json","_metrics.json"), 'w') as fout:
            json.dump(final_metrics, fout, ensure_ascii=False, indent=4)
