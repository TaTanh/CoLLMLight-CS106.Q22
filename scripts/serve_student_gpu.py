import argparse
import json
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import uvicorn

app = FastAPI()

model = None
tokenizer = None

def load_model(base_model_path, adapter_path):
    global model, tokenizer
    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    print(f"Loading base model from {base_model_path} in 4-bit on GPU...")
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print("Model loaded successfully on GPU!")

@app.post("/v1/chat/completions")
def chat_completions(data: dict):
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.1)
    max_new_tokens = data.get("max_tokens", data.get("max_new_tokens", 256))
    
    # Format messages using the tokenizer's chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Move inputs to CUDA
    model_inputs = tokenizer([text], return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True if temperature > 0.05 else False,
            eos_token_id=[151645, 151643],
            pad_token_id=151643
        )
    
    # Strip the input tokens from the generated output
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    
    response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    # Construct response in OpenAI format
    resp = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": response_text
                }
            }
        ]
    }
    return JSONResponse(content=resp)

@app.post("/v1/batch/chat/completions")
def batch_chat_completions(data: dict):
    reqs = data.get("requests", [])
    if not reqs:
        return JSONResponse(content={"responses": []})
        
    temperatures = [r.get("temperature", 0.1) for r in reqs]
    temp = temperatures[0] if temperatures else 0.1
    max_new_tokens = data.get("max_tokens", data.get("max_new_tokens", 256))
    
    texts = []
    for r in reqs:
        messages = r.get("messages", [])
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        texts.append(text)
        
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # Batch generation requires left padding!
    tokenizer.padding_side = "left"
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            temperature=temp,
            do_sample=True if temp > 0.05 else False,
            eos_token_id=[151645, 151643],
            pad_token_id=151643
        )
        
    responses = []
    for i, (input_ids, output_ids) in enumerate(zip(model_inputs.input_ids, generated_ids)):
        gen_tokens = output_ids[model_inputs.input_ids.shape[1]:]
        text_resp = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        responses.append({
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text_resp
                    }
                }
            ]
        })
        
    return JSONResponse(content={"responses": responses})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter", type=str, default="saves/Qwen2.5-1.5B-RCO-LoRA")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    load_model(args.base_model, args.adapter)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
