import sys
import math
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poger-feature', type=str)
    parser.add_argument('--true-prob', type=str)
    parser.add_argument('--output', type=str)
    return parser.parse_args()

def main(args):
    with open(args.poger_feature, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]

    with open(args.true_prob, 'r') as f:
        true_prob = [json.loads(line) for line in f.readlines()]

    for j, item in tqdm(enumerate(data)):
        item['mix_prob_list'] = true_prob[j]['ll_tokens_list']

        item['mix_prob_list'].append([-math.log(1/100)] * len(item['mix_prob_list'][0]))
        for i, idx in enumerate(item['target_prob_idx']):
            item['mix_prob_list'][5][idx] = item['est_prob_list'][5][i]

        item['mix_prob_list'].append([-math.log(1/100)] * len(item['mix_prob_list'][0]))
        for i, idx in enumerate(item['target_prob_idx']):
            item['mix_prob_list'][6][idx] = item['est_prob_list'][6][i]

        del item['est_prob_list']

        with open(args.output, 'a') as f:
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
