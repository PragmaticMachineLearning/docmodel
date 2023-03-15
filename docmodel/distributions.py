from re import S
import fire
import tqdm
import os
from collections import defaultdict

from traitlets import default
from dataset import DocModelDataset
import joblib as jl
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')


def avg_token_length(x):
    ids = x['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(ids)
    n_tokens = len(tokens)
    n_words = len([t for t in tokens if t.startswith('Ġ')]) + 1
    return n_tokens / n_words

def main(
    data_dir=None,
    max_length = 512
):

    dataset = DocModelDataset(
        directory=data_dir,
        split="train",
        max_length=max_length,
        include_filename=True
    )
    input_id_freqs = defaultdict(int)
    
    # Global stats pass
    if not os.path.exists('token_ranks.jl'):
        for i in range(len(dataset.filepaths)):
            try:
                x = dataset[i]
            except:
                continue
            for token_id in x['input_ids'].tolist():
                input_id_freqs[token_id] += 1

        token_counts = sorted(list(input_id_freqs.items()), reverse=True, key=lambda x: x[1])
        token_ranks = {token_id: idx for idx, (token_id, _) in enumerate(token_counts)}
        jl.dump(token_ranks, 'token_ranks.jl')
    else:
        token_ranks = jl.load('token_ranks.jl')

    max_rank = len(token_ranks)
    # Local stats pass

    avg_token_ranks = []
    densities = []
    token_lengths = []
    by_filename = {}

    for i in tqdm.trange(len(dataset.filepaths)):
        try:
            x = dataset[i]
        except:
            continue
        avg_token_rank = np.mean([token_ranks.get(token_id, max_rank) for token_id in x['input_ids'].tolist()])
        avg_token_ranks.append(avg_token_rank)
        points = x['bbox'][:2]
        density = np.mean(pdist(points))
        densities.append(density)
        token_length = avg_token_length(x)
        token_lengths.append(token_length)
        by_filename[x['filename']] = {
            'token_rank': avg_token_rank,
            'density': density,
            'tokens_per_word':token_length
        }

    jl.dump(by_filename, 'stats_by_filename.jl')

    plt.hist(avg_token_ranks)
    plt.savefig('token_rank.png')
    plt.clf()
    plt.hist(densities)
    plt.savefig('densities.png')
    plt.clf()
    plt.hist(token_lengths)
    plt.savefig('token_lengths.png')
    

if __name__ == "__main__":
    fire.Fire(main)
