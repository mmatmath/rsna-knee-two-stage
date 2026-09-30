import json
import os

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer


classes = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]


#----------------------------------------------------------------------------------------------------------------------#
# Configuration                                                                                                        #
#----------------------------------------------------------------------------------------------------------------------#
@dataclass
class Configuration:
    model: str = "Qwen/Qwen3-4B"
    mode: str = "validate"  # "validate" or "full"
    batch_size: int = 4
    max_new_tokens: int = 256
    input_path: str = "./data/train.csv"
    output_path: str = "./data/report_labels_qwen3.csv"


SYSTEM_PROMPT = """You extract strict binary knee MRI findings from a multilingual radiology report.
Read the report in its original language. Return only one valid JSON object, without markdown or explanation.
For every key use 1 when the finding is explicitly present, 0 when it is explicitly absent or negated,
and null when it is uncertain, borderline, or not mentioned. Never invent an unreported finding.

Definitions:
ACL: high-grade partial or complete ACL tear or rupture.
MCL: high-grade acute partial or complete MCL tear.
Medial Meniscus: tear involving the medial meniscus.
Lateral Meniscus: tear involving the lateral meniscus.
Medial OA: definite medial tibiofemoral osteoarthritis with cartilage loss.
Lateral OA: definite lateral tibiofemoral osteoarthritis with cartilage loss.
PF OA: definite patellofemoral osteoarthritis with cartilage loss.
Effusion: moderate or large knee joint effusion.
Synovitis: definite synovitis or synovial proliferation.
Baker's: Baker or popliteal cyst.
Contusion: acute bone contusion or traumatic marrow edema.
Fracture: acute fracture of the imaged knee.

The JSON keys must be exactly: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA,
Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture."""


def parse_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        result = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return {name: result.get(name) if result.get(name) in (0, 1, None) else None for name in classes}


def generate_labels(model, tokenizer, reports, config):
    prompts = []
    for report in reports:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Radiology report:\n" + str(report)},
        ]
        prompts.append(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        ))

    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
        )
    generated = output[:, inputs["input_ids"].shape[1]:]
    return [parse_json(text) for text in tokenizer.batch_decode(generated, skip_special_tokens=True)]


def validate(df):
    rows = []
    for name in classes:
        known = df[name + "_qwen"].notna() & df[name].notna()
        y_true = df.loc[known, name].astype(int)
        y_pred = df.loc[known, name + "_qwen"].astype(int)
        auc = roc_auc_score(y_true, y_pred) if y_true.nunique() == 2 else np.nan
        rows.append({
            "target": name,
            "coverage": known.mean(),
            "accuracy": accuracy_score(y_true, y_pred) if known.any() else np.nan,
            "f1": f1_score(y_true, y_pred, zero_division=0) if known.any() else np.nan,
            "roc_auc": auc,
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main():
    config = Configuration()
    df = pd.read_csv(config.input_path)
    if config.mode == "validate":
        gold = df[classes].notna().all(axis=1)
        df = df.loc[gold].copy()
        output_path = "./data/qwen_gold_predictions.csv"
    else:
        output_path = config.output_path

    done = set()
    previous = pd.DataFrame()
    if os.path.isfile(output_path):
        previous = pd.read_csv(output_path)
        done = set(previous.StudyInstanceUID.astype(str))
    df = df.loc[~df.StudyInstanceUID.astype(str).isin(done)].reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config.model, torch_dtype="auto", device_map="auto",
    ).eval()

    output_rows = []
    for start in range(0, len(df), config.batch_size):
        batch = df.iloc[start:start + config.batch_size]
        labels = generate_labels(model, tokenizer, batch.Report.fillna(""), config)
        for (_, row), label in zip(batch.iterrows(), labels):
            item = {"StudyInstanceUID": row.StudyInstanceUID}
            for name in classes:
                item[name] = None if label is None else label[name]
            output_rows.append(item)
        current = pd.concat([previous, pd.DataFrame(output_rows)], ignore_index=True)
        current.to_csv(output_path, index=False)
        print(f"Processed {min(start + config.batch_size, len(df))}/{len(df)}")

    predictions = pd.concat([previous, pd.DataFrame(output_rows)], ignore_index=True)
    if config.mode == "validate":
        renamed = predictions.rename(columns={name: name + "_qwen" for name in classes})
        validate(pd.read_csv(config.input_path).merge(renamed, on="StudyInstanceUID"))
        print("Saved:", output_path)


if __name__ == "__main__":
    main()
