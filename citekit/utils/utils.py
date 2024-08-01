import numpy as np
import string
import re
import collections
import torch
import nltk

def one_paragraph(text):
    paras = text.lstrip('\n').split('\n\n')
    if not paras:
        return ''
    else:
        return paras[0].rstrip('\n')

def strong_one_paragraph(text):
    paras = text.lstrip('\n').split('\n')
    if not paras:
        return ''
    else:
        return paras[0].rstrip('\n')
    
def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `output`
    Returns:
        STR-EM and STR-EM-HIT ()
    """
    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['short_answers'], item["output"]))
        acc.append(np.mean(loc_acc))
        hit.append(int(np.mean(loc_acc) == 1))

    return 100 * np.mean(acc)
    return 100 * np.mean(acc), 100 * np.mean(hit)

def average(func):
    def avg_func(dataset):
        print(len(dataset))
        results = [func(*data) for data in dataset] if dataset else []
        if results:
            return np.mean(np.array(results), axis=0).tolist()
        else:
            return None
    return avg_func

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_f1(a_gold, a_pred):
    """Compute F1 score between two strings."""

    def _get_tokens(s):
        if not s:
            return []
        return normalize_answer(s).split()

    gold_toks = _get_tokens(a_gold)
    pred_toks = _get_tokens(a_pred)

    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)

    if num_same == 0:
        return 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_exact(a_gold, a_pred):
    """Check whether two strings are equal up to normalization."""

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False

def output_begin_with(word):
    def f(self) -> bool:
        return self.last_message.strip().lower()[:len(word)] == word
    return f

def output_end_with(word):
    def f(self) -> bool:
        return strong_one_paragraph(self.last_message.strip())[-len(word):] == word
    return f


def make_as(datakey):
    def f(passage):
        return {datakey:passage}
    return f

def cut_and_make_as(datakey):
    def f(passage):
        return {datakey:one_paragraph(passage)}
    return f

def remove_citations(sent):
    return re.sub(r"{\d+", "", re.sub(r" {\d+", "", sent)).replace(" |", "").replace("}", "").replace("{", "")

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def match_document(ref_mark, output_ref_span):
    ref = set()
    ref_span = []
    for num in ref_mark:
        ref_str = str(num)
        if ref_str in output_ref_span:
            ref_parts = output_ref_span[ref_str].split("[")
            if len(ref_parts) > 1:
                ref_id_parts = ref_parts[1].split("]")
                if len(ref_id_parts) > 0:
                    ref_id = ref_id_parts[0].strip()
                    if ref_id.isdigit():
                        ref.add(int(ref_id))  # 添加Document id

            ref_span_parts = output_ref_span[ref_str].split(":",1)#第一个冒号后面的片段
            if len(ref_span_parts) > 1:
                ref_span.append(ref_span_parts[1].strip())  # 添加后面的句子片段
            else:
                ref_span.append('')
    return list(ref), ref_span

def get_max_memory():
    """Get the maximum memory available for the current GPU for loading models."""
    free_in_GB = int(torch.cuda.mem_get_info()[0]/1024**3)
    max_memory = f'{free_in_GB-6}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}
    return max_memory

def each_make_as(key):
    def function(output):
        sents = nltk.sent_tokenize(one_paragraph(output))
        if len(sents)>3:
            sents = sents[:3]
        return [make_as(key)(sent) for sent in sents]
    return function

def each_par_make_as(key):
    def function(output):
        sents = one_paragraph(output).split('\n')
        if len(sents)>3:
            sents = sents[:3]
        return [make_as(key)(sent) for sent in sents]
    return function

def sentence(key):
    def function(output):
        sents = nltk.sent_tokenize(one_paragraph(output))
        for sent in sents:
            refs = re.findall(r'\[\d+\]', sent)
            if refs:
                return make_as(key)(sent)
        return make_as(key)('')
    return function

def sentences(key):
    def function(output):
        sents = nltk.sent_tokenize(one_paragraph(output))
        return [make_as(key)(sent) for sent in sents][:1]
    return function

def three_sentences(key):
    def function(output):
        sents = nltk.sent_tokenize(one_paragraph(output))
        return [make_as(key)(sent) for sent in sents][:3]
    return function

def first_sentence(text):
    sents = nltk.sent_tokenize(one_paragraph(text))
    for sent in sents:
        return sent
    return ''

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)



