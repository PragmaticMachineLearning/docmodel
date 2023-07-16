import json
import time


def load_word_freq(freq_file):
    with open(freq_file, "r") as f:
        data = json.load(f)
    return data


WORD_FREQ = load_word_freq("word_freq.json")


def redundancy(text, *args, **kwargs):
    words = text.split()
    if not words:
        return 0
    return len(words) / len(set(words))


def avg_word_length(text, *args, **kwargs):
    words = text.split()
    if not words:
        return 0
    return sum(len(word) for word in words) / len(words)


# What if we had a function that:
# - Takes in a function
# - Returns a function that does everything the original function did
#   but also prints the time it took to run
def timeit(f):
    def modified_f(*args, **kwargs):
        start = time.time()
        output = f(*args, **kwargs)
        end = time.time()
        print(f"Time to run {f.__name__}: {end - start:.3f}")
        return output

    return modified_f


@timeit
def word_freq_per_example(text: str):
    words: list[str] = text.split()

    if not words:
        return 0

    total_words = len(words)
    avg_word_freq = (
        sum(WORD_FREQ[word] for word in words if word in WORD_FREQ) / total_words
    )
    missing_words: list[str] = [word for word in words if word not in WORD_FREQ]
    if missing_words:
        print(f"MISSING WORDS: {''.join(missing_words)}")
    avg_word_freq += len(missing_words) / total_words
    return avg_word_freq
