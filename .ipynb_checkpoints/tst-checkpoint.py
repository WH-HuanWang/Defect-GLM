# -*- encoding: utf-8 -*-

import os
import sys
import torch
import argparse
from transformers import AutoTokenizer
from sat.model.mixins import CachedAutoregressiveMixin
from sat.quantization.kernels import quantize

from model import VisualGLMModel, chat
from finetune_visualglm import FineTuneVisualGLMModel
from sat.model import AutoModel

import json
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int, default=2048, help='max length of the total sequence')
    parser.add_argument("--top_p", type=float, default=0.4, help='top p for nucleus sampling')
    parser.add_argument("--top_k", type=int, default=100, help='top k for top k sampling')
    parser.add_argument("--temperature", type=float, default=.01, help='temperature for sampling')
    parser.add_argument("--english", action='store_true', help='only output English')
    parser.add_argument("--quant", choices=[8, 4], type=int, default=None, help='quantization bits')
    parser.add_argument("--from_pretrained", type=str, default="visualglm-6b", help='pretrained ckpt')
    parser.add_argument("--prompt_zh", type=str, default="描述这张图片。", help='Chinese prompt for the first round')
    parser.add_argument("--prompt_en", type=str, default="What is the defect in this wafer map?", help='English prompt for the first round')
    parser.add_argument("--tst_dataset", type=str, default="finetune_data/dataset_tst.json", help='path of the test dataset json file')
    parser.add_argument("--save_result", type=str, default="results/result.json", help='path of the test dataset json file')
    args = parser.parse_args()

    # load model
    model, model_args = AutoModel.from_pretrained(
        args.from_pretrained,
        args=argparse.Namespace(
        fp16=True,
        skip_init=True,
        use_gpu_initialization=True if (torch.cuda.is_available() and args.quant is None) else False,
        device='cuda' if (torch.cuda.is_available() and args.quant is None) else 'cpu',
    ))
    model = model.eval()

    if args.quant:
        quantize(model.transformer, args.quant)
        if torch.cuda.is_available():
            model = model.cuda()

    model.add_mixin('auto-regressive', CachedAutoregressiveMixin())

    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)

    with open(args.tst_dataset, 'r') as json_file:
        tst_data = json.load(json_file)
        
    sep = 'A:' if args.english else '答：'
    response_list = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(tst_data), total=len(tst_data)):
            dic = {}
            image_path = data["img"]
            history = None
            cache_image = None
            query = args.prompt_en if args.english else args.prompt_zh
            try:
                response, history, cache_image = chat(
                    image_path, 
                    model, 
                    tokenizer,
                    query, 
                    history=history, 
                    image=cache_image, 
                    max_length=args.max_length, 
                    top_p=args.top_p, 
                    temperature=args.temperature,
                    top_k=args.top_k,
                    english=args.english,
                    invalid_slices=[slice(63823, 130000)] if args.english else []
                    )
            except Exception as e:
                print(e)
                break
            
            # print("VisualGLM-6B："+response.split(sep)[-1].strip())
            dic['{}'.format(i)] = response.split(sep)[-1].strip()
            response_list.append(dic)

    with open(args.save_result, 'w') as f:
        json.dump(response_list, f)

if __name__ == "__main__":
    main()