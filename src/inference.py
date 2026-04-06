import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from check_answer import extract_equation


def run_inference(model_path, test_csv_path, output_csv_path):
    df_test = pd.read_csv(test_csv_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    predictions = []

    for _, row in tqdm(df_test.iterrows(), total=len(df_test)):
        task_id = row['id']
        prompt = row['prompt']

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        equation = extract_equation(generated_text)
        if not equation:
            equation = ""

        predictions.append({'id': task_id, 'equation': equation})

    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    run_inference(
        model_path="./gemma-sft",
        test_csv_path="data/raw/test.csv",
        output_csv_path="submissions/sub_v1_sft.csv"
    )