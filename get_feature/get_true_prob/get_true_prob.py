import httpx
import msgpack
import threading
import json
from tqdm import tqdm


def access_api(text, api_url, do_generate=False):
    """
    :param text: input text
    :param api_url: api
    :param do_generate: whether generate or not
    :return:
    """
    with httpx.Client(timeout=None) as client:
        post_data = {
            "text": text,
            "do_generate": do_generate,
        }
        prediction = client.post(api_url,
                                 data=msgpack.packb(post_data),
                                 timeout=None)
    if prediction.status_code == 200:
        content = msgpack.unpackb(prediction.content)
    else:
        content = None
    return content


def get_features(input_file, output_file):
    """
    get [losses, begin_idx_list, ll_tokens_list, label_int, label] based on raw lines
    """

    gpt_2_api = 'http://127.0.0.1:6001/inference'
    gpt_J_api = 'http://127.0.0.1:6002/inference'
    llama_2_api = 'http://127.0.0.1:6003/inference'
    alpaca_api = 'http://127.0.0.1:6004/inference'
    vicuna_api = 'http://127.0.0.1:6005/inference'

    model_apis = [gpt_2_api, gpt_J_api, llama_2_api, alpaca_api, vicuna_api]

    labels = {
        'human': 0,
        'gpt2-xl': 1,
        'gpt-j-6b': 2,
        'Llama-2-13b-chat-hf': 3,
        'vicuna-13b-v1.5': 4,
        'alpaca-7b': 5,
        'gpt-3.5-turbo': 6,
        'gpt-4-1106-preview': 7
    }

    with open(input_file, 'r') as f:
        lines = [json.loads(line) for line in f]

    print('input file:{}, length:{}'.format(input_file, len(lines)))

    with open(output_file, 'w', encoding='utf-8') as f:
        for data in tqdm(lines):
            line = data['text']
            label = data['label']

            losses = []
            begin_idx_list = []
            ll_tokens_list = []
            model_apis = model_apis
            label_dict = labels

            label_int = label_dict[label]

            error_flag = False
            for api in model_apis:
                try:
                    loss, begin_word_idx, ll_tokens = access_api(line, api)
                except TypeError:
                    print("return NoneType, probably gpu OOM, discard this sample")
                    error_flag = True
                    break
                losses.append(loss)
                begin_idx_list.append(begin_word_idx)
                ll_tokens_list.append(ll_tokens)
            # if oom, discard this sample
            if error_flag:
                continue

            result = {
                'losses': losses,
                'begin_idx_list': begin_idx_list,
                'll_tokens_list': ll_tokens_list,
                'label_int': label_int,
                'label': label,
                'text': line
            }

            f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    input_files = ['../../data/train.jsonl',
                   '../../data/val.jsonl',
                   '../../data/test.jsonl']

    output_files = ['./result/train_true_prob.jsonl',
                   './result/val_true_prob.jsonl',
                   './result/test_true_prob.jsonl']

    threads = []
    for i in range(len(input_files)):
        t = threading.Thread(target=get_features, args=(input_files[i], output_files[i]))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
