import os
import sys
import json
import math
import nltk
import torch
import argparse
import tiktoken
import numpy as np
import concurrent.futures

from tqdm import tqdm
from openai import OpenAI
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer, AutoModelForCausalLM, RobertaTokenizer


def get_root_word(word):
    from nltk.stem import SnowballStemmer
    stemmer = SnowballStemmer("english")
    return stemmer.stem(word)


class GPT2Estimator:

    def __init__(self, model_name_or_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, pad_token_id=self.tokenizer.eos_token_id).to(device)

    def estimate(self, context, n, temp, max_token):
        input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=max_token, do_sample=True, temperature=temp, num_return_sequences=n)
        output = self.tokenizer.batch_decode(generate_ids[:, len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        cnt = {"OTHER": 0}
        for item in output:
            try:
                word_list = nltk.word_tokenize(item)
                first_word = [word for word in word_list if word.isalnum()][0]
                root_word = get_root_word(first_word)
                if root_word in cnt:
                    cnt[root_word] += 1
                else:
                    cnt[root_word] = 1
            except:
                cnt["OTHER"] += 1
        cnt
        return cnt


class GPTJEstimator:

    def __init__(self, model_name_or_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, pad_token_id=self.tokenizer.eos_token_id).to(device)

    def estimate(self, context, n, temp, max_token):
        input_ids = self.tokenizer(context, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=max_token, do_sample=True, temperature=temp, num_return_sequences=n)
        output = self.tokenizer.batch_decode(generate_ids[:, len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        cnt = {"OTHER": 0}
        for item in output:
            try:
                word_list = nltk.word_tokenize(item)
                first_word = [word for word in word_list if word.isalnum()][0]
                root_word = get_root_word(first_word)
                if root_word in cnt:
                    cnt[root_word] += 1
                else:
                    cnt[root_word] = 1
            except:
                cnt["OTHER"] += 1
        cnt
        return cnt


class LLaMA2Estimator:

    def __init__(self, model_name_or_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, use_safetensors=False).half().to(device)

    def get_prompt(self, message, chat_history, system_prompt):
        texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n[/INST]\n']
        # The first user input is _not_ stripped
        do_strip = False
        for user_input, response in chat_history:
            user_input = user_input.strip() if do_strip else user_input
            do_strip = True
            texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
        message = message.strip() if do_strip else message
        texts.append(f'{message}')
        return ''.join(texts)

    def estimate(self, context, n, temp, max_token):
        system_prompt = "Please continue writing the following text in English, starting from the next word and not repeating existing content."
        prompt = self.get_prompt(context, [], system_prompt)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=max_token, do_sample=True, temperature=temp, num_return_sequences=n)
        output = self.tokenizer.batch_decode(generate_ids[:, len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        cnt = {"OTHER": 0}
        for item in output:
            try:
                word_list = nltk.word_tokenize(item)
                first_word = [word for word in word_list if word.isalnum()][0]
                root_word = get_root_word(first_word)
                if root_word in cnt:
                    cnt[root_word] += 1
                else:
                    cnt[root_word] = 1
            except:
                cnt["OTHER"] += 1
        cnt
        return cnt


class AlpacaEstimator:

    def __init__(self, model_name_or_path, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

    def get_prompt(self, instruction, input):
        return f'### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n'

    def estimate(self, context, n, temp, max_token):
        instruction = "Please continue writing the following text in English, starting from the next word and not repeating existing content."
        prompt = self.get_prompt(instruction, context)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=max_token, do_sample=True, temperature=temp, num_return_sequences=n)
        output = self.tokenizer.batch_decode(generate_ids[:, len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        cnt = {"OTHER": 0}
        for item in output:
            try:
                word_list = nltk.word_tokenize(item)
                first_word = [word for word in word_list if word.isalnum()][0]
                root_word = get_root_word(first_word)
                if root_word in cnt:
                    cnt[root_word] += 1
                else:
                    cnt[root_word] = 1
            except:
                cnt["OTHER"] += 1
        cnt
        return cnt


class VicunaEstimator:

    def __init__(self, model_name_or_path, device):
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).half().to(device)

    def get_prompt(self, message):
        conv = get_conversation_template(self.model_name_or_path)
        conv.append_message(conv.roles[0], message)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt

    def estimate(self, context, n, temp, max_token):
        instruction = "Please continue writing the following text in English, starting from the next word and not repeating existing content: "
        prompt = self.get_prompt(instruction + context)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=max_token, do_sample=True, temperature=temp, num_return_sequences=n)
        output = self.tokenizer.batch_decode(generate_ids[:, len(input_ids[0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        cnt = {"OTHER": 0}
        for item in output:
            try:
                word_list = nltk.word_tokenize(item)
                first_word = [word for word in word_list if word.isalnum()][0]
                root_word = get_root_word(first_word)
                if root_word in cnt:
                    cnt[root_word] += 1
                else:
                    cnt[root_word] = 1
            except:
                cnt["OTHER"] += 1
        cnt
        return cnt


class GPT35TurboEstimator:

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def estimate(self, context, n, temp, max_token):
        max_try = 5
        cnt = {"OTHER": 0}
        remaining = n
        while remaining > 0:
            for _ in range(max_try):
                try:
                    completion = self.client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "Please continue writing the following text in English, starting from the next word and not repeating existing content."},
                            {"role": "user", "content": context}
                        ],
                        temperature=temp,
                        max_tokens=max_token,
                        n=min(remaining, 128)
                    )

                    for choice in completion.choices:
                        try:
                            word_list = nltk.word_tokenize(choice.message.content)
                            first_word = [word for word in word_list if word.isalnum()][0]
                            root_word = get_root_word(first_word)
                            if root_word in cnt:
                                cnt[root_word] += 1
                            else:
                                cnt[root_word] = 1
                        except:
                            cnt["OTHER"] += 1
                    remaining -= min(remaining, 128)
                    break
                except Exception as e:
                    print(e)
        return cnt


class GPT4TurboEstimator:

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def estimate(self, context, n, temp, max_token):
        max_try = 5
        cnt = {"OTHER": 0}
        remaining = n
        while remaining > 0:
            for _ in range(max_try):
                try:
                    completion = self.client.chat.completions.create(
                        model="gpt-4-1106-preview",
                        messages=[
                            {"role": "system", "content": "Please continue writing the following text in English, starting from the next word and not repeating existing content."},
                            {"role": "user", "content": context}
                        ],
                        temperature=temp,
                        max_tokens=max_token,
                        n=min(remaining, 128)
                    )

                    for choice in completion.choices:
                        try:
                            word_list = nltk.word_tokenize(choice.message.content)
                            first_word = [word for word in word_list if word.isalnum()][0]
                            root_word = get_root_word(first_word)
                            if root_word in cnt:
                                cnt[root_word] += 1
                            else:
                                cnt[root_word] = 1
                        except:
                            cnt["OTHER"] += 1
                    remaining -= min(remaining, 128)
                    break
                except Exception as e:
                    print(e)
        return cnt


estimators = [
    GPT2Estimator('gpt2-xl', device=torch.device('cuda:3')),
    GPTJEstimator('EleutherAI/gpt-j-6b', device=torch.device('cuda:3')),
    LLaMA2Estimator('meta-llama/Llama-2-13b-chat-hf', token=os.environ['HF_TOKEN'], device=torch.device('cuda:2')),
    AlpacaEstimator('/path/to/alpaca-7b/', device=torch.device('cuda:1')),
    VicunaEstimator('lmsys/vicuna-13b-v1.5', device=torch.device('cuda:0')),
    GPT35TurboEstimator(os.environ['OPENAI_API_KEY']),
    GPT4TurboEstimator(os.environ['OPENAI_API_KEY'])
]

roberta_tknz = RobertaTokenizer.from_pretrained("roberta-base")

label2id = {
    'human': 0,
    'gpt2-xl': 1,
    'gpt-j-6b': 2,
    'Llama-2-13b-chat-hf': 3,
    'vicuna-13b-v1.5': 4,
    'alpaca-7b': 5,
    'gpt-3.5-turbo': 6,
    'gpt-4-1106-preview': 7
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--delta', type=float, default=1.2)
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    return parser.parse_args()


def get_estimate_prob(context, target_word, n, temp, max_token):
    ans = []
    target_word = get_root_word(target_word)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(estimator.estimate, context, n, temp, max_token) for estimator in estimators]
        for future in concurrent.futures.as_completed(futures):
            cnt = future.result()
            if target_word in cnt:
                ans.append(-1 * math.log(cnt[target_word] / sum(cnt.values())))
            else:
                ans.append(-1 * math.log(1/n))
    return ans


def get_word_list(text):
    word_list = nltk.word_tokenize(text)

    for i, word in enumerate(word_list):
        if len(word) < 2 and word not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            word_list[i] = "[SKIP]"
        if word in ["``", "''"]:
            word_list[i] = "[SKIP]"

    pos = []
    word_start = 0
    for word in word_list:
        if word == "[SKIP]":
            pos.append(word_start)
            continue

        while text[word_start] != word[0]:
            word_start += 1
        pos.append(word_start)
        word_start += len(word)

    return word_list, pos


def calc_max_logprob(n, delta):
    min_p = 1 / (1 + n / (1.96/delta)**2)
    max_logp = -math.log(min_p)
    return max_logp


def main(args):
    max_loss = calc_max_logprob(args.n, args.delta)
    enc = tiktoken.get_encoding("cl100k_base")

    with open(args.input, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

    for item in tqdm(data):
        item['label_int'] = label2id[item['label']]
        word_list, pos = get_word_list(item['text'])
        proxy_ll = list(zip(range(len(word_list)), item['proxy_prob'], word_list, pos))

        skip_prefix = 20
        proxy_ll = proxy_ll[skip_prefix:]
        proxy_ll = [item for item in proxy_ll if item[2] != "[SKIP]"]

        proxy_ll = [item for item in proxy_ll if item[1] <= max_loss]

        proxy_ll = sorted(proxy_ll, key=lambda x: x[1], reverse=True)[:args.k]
        proxy_ll = sorted(proxy_ll, key=lambda x: x[0])

        est_prob = []
        for proxy_ll_item in proxy_ll:
            context = item['text'][:proxy_ll_item[3]]
            context = enc.decode(enc.encode(context)[-20:])
            est_prob.append(get_estimate_prob(context, proxy_ll_item[2], args.n, 1.5, 2))
        est_prob = np.array(est_prob)
        est_prob = est_prob.T
        item['est_prob_list'] = est_prob.tolist()

        prefix_text_list = [item['text'][:proxy_ll_item[3]] for proxy_ll_item in proxy_ll]
        target_roberta_idx_list = [len(ids)-2 for ids in roberta_tknz(prefix_text_list).input_ids]
        item['target_roberta_idx'] = target_roberta_idx_list

        item['target_prob_idx'] = [proxy_ll_item[3] for proxy_ll_item in proxy_ll]
        del item['proxy_prob']
        del item['source']

        with open(args.output, 'a') as f:
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
