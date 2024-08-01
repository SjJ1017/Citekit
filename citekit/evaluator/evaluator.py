from nltk import sent_tokenize
import nltk
nltk.download('punkt')
import re 
import random
import transformers
import numpy as np
from citekit.utils.utils import *
from rouge import Rouge
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import copy
import torch
from tqdm import tqdm
import sys
import logging
import random
from itertools import product,combinations
import time
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PIPELINE_OUTPUT = 'output'
PIPELINE_DOC_CACHE = 'doc_cache'

global autoais_model, autoais_tokenizer
autoais_model = None
autoais_tokenizer = None
get_docs_by_index = lambda i,docs: docs[i] if i < len(docs) else None 
ais_LLM = None

QA_MODEL = "gaotianyu1350/roberta-large-squad"
AUTOAIS_MODEL = "google/t5_xxl_true_nli_mixture"


def get_cite(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", ""),[int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]


def entail(premise, claim):

    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    global autoais_model, autoais_tokenizer
    input_text = "premise: {} hypothesis: {}".format(premise, claim)
    input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
    with torch.inference_mode():
        outputs = autoais_model.generate(input_ids, max_new_tokens=10)
    result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
    inference = 1 if result == "1" else 0
    return inference

def load_auto_ais():
    global autoais_model, autoais_tokenizer
    print('Initializing eval model for citation precision and recall...') 
    autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
    print('Done!')

def compute_mauve(data):
    """Compute Mauve score."""

    logger.info("Computing MAUVE...")
    human_data = []
    model_data = []
    for item in data:
        # Remove ending punctuations
        # Remove any new lines
        # Truncate by 100 words
        human_data.append(
            ' '.join((item['question'] + " " + item['answer'].strip()).split()[:100]).rstrip(string.punctuation))
        model_data.append(
            ' '.join((item['question'] + " " + item['output'].strip()).split()[:100]).rstrip(string.punctuation))

    import mauve
    out = mauve.compute_mauve(
        p_text=human_data,
        q_text=model_data,
        device_id=0,
        max_text_length=512,
        verbose=True,
        batch_size=8,
        featurize_model_name="gpt2-large"
    )
    return out.mauve * 100


def compute_rouge_l(data):
    total = len(data)
    res = {
                "r": 0.0,
                "p": 0.0,
                "f": 0.0
            }
    for item in data:
        if item['output'] and item['answer']:
            rouge = Rouge()
            scores = rouge.get_scores(item['output'], item['answer'])
            res['r'] += scores[0]['rouge-l']['r']
            res['p'] += scores[0]['rouge-l']['p']
            res['f'] += scores[0]['rouge-l']['f']
        else:
            print('Warning: no hypothesis or references')
    res['r'] /= total
    res['p'] /= total
    res['f'] /= total

    return res
    
def compute_qa(question, output, short_answers, qa_pipeline=None):
    """Compute QA-based accuracy.
    Args:
        
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """

    # Load model
    if not qa_pipeline:
        qa_pipeline = transformers.pipeline("question-answering", model=QA_MODEL, device='mps')

    # Get prediction
    em, f1, bins = 0,0,0
    context = output if len(output) > 0 else " "
    result = qa_pipeline(question=question, context=context, handle_impossible_answer=True)
    loc_counter, loc_em, loc_f1 = 0, 0, 0
    print(result)
    prediction = result["answer"]

    loc_em = max([compute_exact(a, prediction) for a in short_answers])
    loc_f1 = max([compute_f1(a, prediction) for a in short_answers])
    loc_counter += 1

    em= loc_em / loc_counter
    f1= loc_f1 / loc_counter 
    bins = int(loc_em == loc_counter)
    return em, f1, bins

def compute_qa(data):
    """Compute QA-based accuracy.
    Args:
        data: requires filed `qa_pairs/short_answers` and `output`
    Returns:
        QA metrics (QA-EM, QA-F1, QA-Hit)
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        #logger.warn("Warning: no QA pairs found in data")
        return {
            'QA-EM': 0,
            'QA-F1': 0,
            'QA-Hit': 0,
        }

    # Load model
    #logger.info("Loading the RoBERTa-large SQuAD model for QA-based accuracy...")
    global qa_pipeline
    if not qa_pipeline:
        qa_pipeline = transformers.pipeline("question-answering", model=QA_MODEL)
    #logger.info("Done")

    # Get prediction
    #logger.info("Computing the QA-based accuracy...")
    em, f1, bins = [], [], []
    for item in tqdm(data):
        question = [qa_pair['question'] for qa_pair in item['qa_pairs']]
        context = item['output'] if len(item['output']) > 0 else " "
        results = qa_pipeline(question=question, context=context, handle_impossible_answer=True)
        loc_counter, loc_em, loc_f1 = 0, 0, 0

        for idx, res in enumerate(results):
            answers = item["qa_pairs"][idx]["short_answers"]
            prediction = res["answer"]

            loc_em += max([compute_exact(a, prediction) for a in answers])
            loc_f1 += max([compute_f1(a, prediction) for a in answers])
            loc_counter += 1

        em.append(loc_em / loc_counter)
        f1.append(loc_f1 / loc_counter)
        bins.append(loc_em == loc_counter)

    return {
        'QA-EM': 100 * np.mean(em),
        'QA-F1': 100 * np.mean(f1),
        'QA-Hit': 100 * np.mean(bins)
    }


def cite_pr(sent_with_cite, docs = None, get_docs = get_docs_by_index, get_cite = get_cite, max_cite= None,rich_return = False):
    """
    : sent_with_cite: ONE sentence with citation like [1][2][3]
    : get_docs: by default like [1][2], get ids
    : docs: List, all the COMPLETE documents with TITLE

    : return 
        number of citations, integer
        recall (0 or 1)
        precision (number of relevent documents)

        optional;
            multi_cite
            mcite_support
            mcite_overcite
    """
    if rich_return:
        raise NotImplementedError

    result = {'num_cites': 0,'recall':0,'precision':0,'multi_cite':0,'mcite_support' :0,'mcite_overcite':0}
    sent, cites= get_cite(sent_with_cite)

    if not cites:
        return (0, 0, 0) if not rich_return else result # no citations
    if max_cite:
        cites = cites[:max_cite]
    num_cites = len(cites)
    result['num_cites'] = num_cites

    refs = [get_docs(cite, docs) for cite in cites]
    if None in refs:
        return (num_cites, 0, 0) if not rich_return else result# wrong citation(s)
    
    # recall
    recall = entail(premise=''.join(refs),claim=sent)
    result['recall'] = recall

    # precision
    precision = 0
    if num_cites == 1:
        precision = recall
    else:
        for idx, ref in enumerate(refs):
            if entail(premise=ref,claim=sent):
                precision += 1
            else:
                if not entail(premise=''.join([refs[i] for i in range(len(refs)) if i != idx]), claim = sent):
                    precision += 1
                elif recall:
                    result['mcite_overcite'] = 1
    result['precision'] = precision
    
    #other 
    if num_cites > 1:
        result['multi_cite'] = 1
        if recall:
            result['mcite_support'] = 1
    

    return (num_cites, recall, precision) if not rich_return else result


def cite_pr_answer(answer, docs = None, get_docs = get_docs_by_index, get_cite = get_cite, max_cite= None,rich_return = False):
    epsilon = 1e-8
    num_c = 0
    recall = 0
    precision = 0
    sents = sent_tokenize(answer)
    for sent in sents:
        c,r,p = cite_pr(sent,get_docs=get_docs,docs=docs,get_cite=get_cite,max_cite=max_cite,rich_return=rich_return)
        num_c += c
        recall += r
        precision += p
    # diveded by Zero!
    return recall/(len(sents)+ epsilon), precision/(num_c+epsilon)


def _run_nli_autoais(passage, claim, test = False):
    """
    Run inference for assessing AIS between a premise and hypothesis.
    Adapted from https://github.com/google-research-datasets/Attributed-QA/blob/main/evaluation.py
    """
    if not test:
        global autoais_model, autoais_tokenizer
        if not autoais_model:
            load_auto_ais()
        input_text = "premise: {} hypothesis: {}".format(passage, claim)
        input_ids = autoais_tokenizer(input_text, return_tensors="pt").input_ids.to(autoais_model.device)
        with torch.inference_mode():
            outputs = autoais_model.generate(input_ids, max_new_tokens=10)
        result = autoais_tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference = 1 if result == "1" else 0
        return inference
    else:
        res = random.randint(0,1)

    return res

def _run_llm_autoais(passage, claim):
    global ais_LLM
    assert(ais_LLM)
    return int(ais_LLM.generate(premise = passage, claim = claim))

def test_compute_autoais(data):
    print(data[0]['docs'][:5])
    print(data[0]['output'][:5])
    return {
        "citation_rec": random.randint(0,100),
        "citation_prec": random.randint(0,100),
    }

def compute_autoais(data,
                    decontext=False,
                    concat=False,
                    qampari=False,
                    at_most_sents = 3,
                    at_most_citations=3,
                    entail_function = _run_nli_autoais):
    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """

    global autoais_model, autoais_tokenizer


    ais_scores = []
    ais_scores_prec = []

    sent_total = 0
    sent_mcite = 0
    sent_mcite_support = 0
    sent_mcite_overcite = 0
    autoais_log = []
    for item in tqdm(data):
        # Get sentences by using NLTK
        if qampari:
            print('now qampari...')
            sents = [item['question'] + " " + x.strip() for x in
                     item['output'].rstrip().rstrip(".").rstrip(",").split(",")]
        else:
            sents = sent_tokenize(item['output'])[:at_most_sents]
        if len(sents) == 0:
            ais_scores.append(0.0)
            ais_scores_prec.append(0.0)  # len(sents))
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]

        entail = 0
        entail_prec = 0
        total_citations = 0
        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id]  # Citation removed and (if opted for) decontextualized
            joint_entail = -1  # Undecided

            # Find references
            #ref = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]  # In text citation id starts from 1
            matches = re.findall(r"\[(\d+(?:,\s*\d+)*)\]", sent)
            ref = [int(num)-1 for match in matches for num in match.replace(' ', '').split(',')]
            if len(ref) == 0:
                # No citations
                joint_entail = 0
            elif any([ref_id >= len(item['docs']) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)
                joint_passage = '\n'.join([(item['docs'][psgs_id]) for psgs_id in ref])

            # If not directly rejected by citation format error, calculate the recall score
            if joint_entail == -1:
                joint_entail = entail_function(joint_passage, target_sent)
                autoais_log.append({
                    #"question": item['question'],
                    "output": item['output'],
                    "claim": sent,
                    "passage": [joint_passage],
                    "model_type": "NLI",
                    "model_output": joint_entail,
                })

            entail += joint_entail
            if len(ref) > 1:
                sent_mcite += 1

            # calculate the precision score if applicable
            if joint_entail and len(ref) > 1:
                sent_mcite_support += 1
                # Precision check: did the model cite any unnecessary documents?
                for psgs_id in ref:
                    # condition A
                    passage = item['docs'][psgs_id]
                    nli_result = entail_function(passage, target_sent)

                    # condition B
                    if not nli_result:
                        subset_exclude = copy.deepcopy(ref)
                        subset_exclude.remove(psgs_id)
                        passage = '\n'.join([item['docs'][pid] for pid in subset_exclude])
                        nli_result =entail_function(passage, target_sent)
                        if nli_result:  # psgs_id is not necessary
                            flag = 0
                            sent_mcite_overcite += 1
                        else:
                            entail_prec += 1
                    else:
                        entail_prec += 1
            else:
                entail_prec += joint_entail
        sent_total += len(sents)
        ais_scores.append(entail / len(sents))
        ais_scores_prec.append(entail_prec / total_citations if total_citations > 0 else 0)  # len(sents))

    if sent_mcite > 0 and sent_mcite_support > 0:
        print(
            "Among all sentences, %.2f%% have multiple citations, among which %.2f%% are supported by the joint set, among which %.2f%% overcite." % (
                100 * sent_mcite / sent_total,
                100 * sent_mcite_support / sent_mcite,
                100 * sent_mcite_overcite / sent_mcite_support
            ))

    return {
        "citation_rec": 100 * np.mean(ais_scores),
        "citation_prec": 100 * np.mean(ais_scores_prec),
    }

def compute_claims_test(data):
    print(data[0]['claims'])
    print(data[0][PIPELINE_OUTPUT])
    return random.randint(1,100)

def compute_claims(data):
    global autoais_model, autoais_tokenizer
    if autoais_model is None:
        #logger.info("Loading AutoAIS model...")
        # autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16,
                                                              device_map="auto")
        # autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto",offload_folder= "/data/hongbang/zsf/projects/ALCE/ALCE/model/t5_xxl_true_nli_mixture/offload1")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
    #logger.info("Computing claims...")
    scores = []
    for item in tqdm(data):
        normalized_output = remove_citations(item['output'])
        entail = 0
        claims = item["claims"]
        for claim in claims:
            entail += _run_nli_autoais(normalized_output, claim)
        scores.append(entail / len(claims))
    return 100 * np.mean(scores)


#citation appropriateness
def check_if_citations_needed(passages, answer, grain):

    def _format_document(doc):
        """Format document for AutoAIS.

        if "sent" in doc:
            # QA-extracted docs
            return "Title: %s\n%s" % (doc['title'], doc['sent'])
        else:
            return "Title: %s\n%s" % (doc['title'], doc['text'])
        """
        return doc

    global autoais_model, autoais_tokenizer
    if autoais_model is None and False:
        #logger.info("Loading AutoAIS model...")
        # autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16,
                                                              device_map="auto")
        # autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto",offload_folder= "/data/hongbang/zsf/projects/ALCE/ALCE/model/t5_xxl_true_nli_mixture/offload1")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)

    if grain=="over_fine" or grain=="more_over_fine":
        num_passages = len(passages)
        passages_per_chunk = num_passages // 5  # Divide passages evenly into 5 chunks
        remainder = num_passages % 5  # Handle remaining passages
        passages_five=[]
        start_idx = 0
        for i in range(5):
            end_idx = start_idx + passages_per_chunk
            if remainder > 0:
                end_idx += 1
                remainder -= 1
            chunk_passages = passages[start_idx:end_idx]
            passages_five.append('\n'.join([_format_document(p) for p in chunk_passages]))
            start_idx = end_idx
        passages=passages_five
        combinations_3 = combinations(passages, 3)  # 获取所有三个passage的组合
        for combination in combinations_3:
            joint_passage = '\n'.join(
                [passage for passage in combination])  # 将三个passage连接为一个字符串，并保留格式
            entail = _run_nli_autoais(joint_passage, answer)
            if entail == 1:
                return 1
        return 0

    else:
        if len(passages)>=3:#正常粒度
            combinations_3 = combinations(passages, 3)
            for combination in combinations_3:
                joint_passage = '\n'.join(
                    [_format_document(passage) for passage in combination])
                entail = _run_nli_autoais(joint_passage, answer)
                if entail == 1:
                    return 1
            return 0
        else:#粗粒度
            joint_passage = '\n'.join(
                [_format_document(passage) for passage in passages])
            entail = _run_nli_autoais(joint_passage, answer)
            if entail == 1:
                return 1
            else:
                return 0


#citaion granularity
def find_permutations(n, m):
    '''
    :param n:  最大数量总和
    :param m: 位长度
    :return:
    '''
    # Generate all possible sequences of length m
    all_sequences = list(product(range(n + 1), repeat=m))
    #print('all_sequences', all_sequences)

    # Filter sequences where the sum of digits equals n
    valid_sequences = [seq for seq in all_sequences if sum(seq) == n]
    return valid_sequences


def get_subspans(list_span, span_count):
    list_subspan = []
    for i in range(0, len(list_span) - span_count + 1):
        list_subspan.append(list_span[i: i + span_count])

    return list_subspan


def get_all_span_comb(list_list_span, target_span_count=-1):
    if target_span_count == -1: # 所有子集
        max_span_count = len(sum(list_list_span, []))
        doc_count = len(list_list_span)
        list_span_comb_all = []
        for span_count in range(1, max_span_count + 1):
            list_comb = find_permutations(span_count, doc_count)#给定数量的子串在文本中的所有可能组合

            list_span_comb = [] # 最终当前长度的所有可能组合
            for comb in list_comb:
                list_list_subspan = []

                for idx_doc, span_count_doc in enumerate(comb):
                    list_subspan = get_subspans(list_list_span[idx_doc], span_count_doc)
                    if len(list_subspan) == 0:
                        list_list_subspan = None
                        break
                    list_list_subspan.append(list_subspan)

                if list_list_subspan:
                    list_span_comb_cur = [sum(list(combination), [])  for combination in product(*list_list_subspan)]
                    list_span_comb_cur = list(set([tuple(span_comb) for span_comb in list_span_comb_cur]))

                    list_span_comb += list_span_comb_cur
            list_span_comb_all += list_span_comb
        list_span_comb_all = set(list_span_comb_all)
    else: # 当前长度的组合
        doc_count = len(list_list_span)
        list_comb = find_permutations(target_span_count, doc_count)

        list_span_comb = []  # 最终当前长度的所有可能组合
        for comb in list_comb:
            list_list_subspan = []

            for idx_doc, span_count_doc in enumerate(comb):
                list_subspan = get_subspans(list_list_span[idx_doc], span_count_doc)
                if len(list_subspan) == 0:
                    list_list_subspan = None
                    break
                list_list_subspan.append(list_subspan)

            if list_list_subspan:
                list_span_comb_cur = [combination for combination in product(*list_list_subspan)]
                for idx in range(len(list_span_comb_cur)):
                    list_span_comb_cur[idx] = tuple([tuple(span_comb) for span_comb in list_span_comb_cur[idx]])

                list_span_comb += list_span_comb_cur
        list_span_comb_all = list_span_comb
        list_span_comb_all = set(list_span_comb_all)
    return list_span_comb_all


def run_converge_2(list_list_span=None, sentence=None):
    '''
    基于假设：更长的text不能蕴含，则其任何子串都不能蕴含
    span数量递减（提供更多的剪枝选项）
    最终gold可能有一个span的误差
    '''
    ######
    #print('origin nli count', len(get_all_span_comb(list_list_span, target_span_count=-1)))#给定文本的所有可能的子串组合
    max_span_count = len(sum(list_list_span, [])) # span总数

    set_comb_hash = set([])

    ### span数量二分
    nli_count = 0
    skip_count = 0
    list_list_span_gold = copy.copy(list_list_span) # 当前能够精准蕴含的span

    span_count_min, span_count_max = 1, max_span_count
    start_time=time.time()
    timeout=300
    while span_count_min < span_count_max:#每次迭代中不断寻找更小的子串组合
        span_count_cur = span_count_max - 1
        flag_find = False
        if time.time() - start_time > timeout:
            print('timeout!')
            list_list_span_gold=[]
            break
        ### 存在可蕴含，继续找更少的span
        ### 不存在可蕴含，继续找更多的span
        # 长度为span_count_max - 1的所有可能的子串组合
        set_comb_cur = get_all_span_comb(list_list_span, target_span_count=span_count_cur)

        list_comb_cur = list(set_comb_cur)
        random.shuffle(list_comb_cur)
        for comb in list_comb_cur:
            list_list_span_cur = [list(t) for t in comb]
            list_span_cur = sum(list_list_span_cur, [])
            str_text = ' '.join(list_span_cur) # TODO: 统一字符串化的方式

            if hash(str_text) in set_comb_hash:
                skip_count += 1
                continue

            #### ⚠️ 注意在这里替换nli函数
            nli_label = _run_nli_autoais(str_text, sentence) # TODO: nli label function
            nli_count += 1

            if nli_label == 1: # 只要存在可蕴含，直接继续找更少的span
                list_list_span_gold = copy.copy(list_list_span_cur)
                span_count_max = span_count_cur#更新span数量上限
                flag_find = True
                # print(f"find nli!, nli_count: {nli_count}, skip_count: {skip_count}, len(set_comb_hash): {len(set_comb_hash)}", )
                break
            else: # 不能蕴含，剪枝所有子集
                set_comb_cur_del = get_all_span_comb(list_list_span_cur, target_span_count=-1)
                set_comb_hash_cur = set([hash(' '.join(list(tuple_comb_))) for tuple_comb_ in set_comb_cur_del]) # TODO: 统一字符串化的方式

                set_comb_hash |= set_comb_hash_cur
        if flag_find == False:
            print(f"CAN'T find nli!, nli_count: {nli_count}, skip_count: {skip_count}, len(set_comb_hash): {len(set_comb_hash)}", )
            break
    span_count_gold = span_count_max # gold的span数量
    print('len(set_comb_del)', len(set_comb_hash))
    print('nli_count', nli_count, 'skip_count', skip_count, 'span_count_gold', span_count_gold)
    return list_list_span_gold


def compute_autoais_grained(data,
                    at_most_citations=3,method='ALCE',grain='default'):

    """
    Compute AutoAIS score.

    Args:
        data: requires field `output` and `docs`
              - docs should be a list of items with fields `title` and `text` (or `phrase` and `sent` for QA-extracted docs)
        citation: check citations and use the corresponding references.
        decontext: decontextualize the output
    """
    global autoais_model, autoais_tokenizer
    if autoais_model is None and False:
        #logger.info("Loading AutoAIS model...")
        # autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto")
        autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16,
                                                              device_map="auto")
        # autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, max_memory=get_max_memory(), device_map="auto",offload_folder= "/data/hongbang/zsf/projects/ALCE/ALCE/model/t5_xxl_true_nli_mixture/offload1")
        autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
    def _format_document(doc):

        """Format document for AutoAIS."""
        if isinstance(doc, dict):
            if "sent" in doc:
                # QA-extracted docs
                return "Title: %s\n%s" % (doc['title'], doc['sent'])
            else:
                return "Title: %s\n%s" % (doc['title'], doc['text'])
        elif isinstance(doc,str):
            return doc


    #logger.info(f"Running AutoAIS...")

    ais_scores_need = []  # 是否需要引用
    ais_scores = []  # quote_recall
    ais_doc_scores=[]#doc_recall

    sent_total = 0

    autoais_log = []
    granularity_list = []
    skipped =0
    for item in tqdm(data):
        output = item['output']

        if method=='baseline':
            model_answer=item['output_parse']['answer']
            answer = ''
            reference = {}
            span_contents = {}
            if not model_answer["text"].endswith("."):
                model_answer["text"] += "."
            answer += " " + model_answer["text"]
            spans = model_answer['reference']
            for span in spans:
                match = re.match(r'^(\d+)\.', span)
                if match:
                    span_number = match.group(1)
                    span_content = span.split('. ', 1)[1].strip()  # 获取1. 后面的内容
                    span_contents[span_number] = span_content
            reference.update(span_contents)

            item['output_answer'] = answer.strip()
            item['output_ref_span'] = reference
            output = item['output_answer']

        elif method=='ALCE':
            # 匹配 According to Document
            pattern_doc = r"According to Document \[(\d+)\]"
            # 匹配 (Title: Godfrey Chitalu)
            pattern_title = r"\(Title: [^\)]+\)"

            output = re.sub(pattern_doc, r"[\1]", output)
            output = re.sub(pattern_title, "", output)
            output=output.strip().split("\n")[0]
            output=output.replace("<|im_end|>", "")
        # Get sentences by using NLTK
        sents = sent_tokenize(output)[:3]
        if len(sents) == 0:
            continue

        target_sents = [remove_citations(sent).strip() for sent in sents]
        output_ref_span = item.get('output_ref_span', {})
        # sent_joint_passage = '\n'.join([_format_document(doc) for doc in item['docs']])

        entail = 0
        entail_doc=0
        total_citations = 0
        need_citations_sentences = 0  # 一个回答中需要引用的句子数量
        correct_predictions = 0  # 新增：记录正确的预测是否需要引用的子句数量

        for sent_id, sent in enumerate(sents):
            target_sent = target_sents[sent_id]  # Citation removed and (if opted for) decontextualized
            joint_entail = -1  # Undecided
            joint_doc_entail=-1

            # 1. appropriatness
            # 每句话是否需要引用
            need_citations = check_if_citations_needed(item['docs'], target_sent,grain)


            if method=='baseline':
                # Find references number
                ref_mark = [int(r[1:]) for r in re.findall(r"\{\d+", sent)]
                # 引用的span（拼接）match document
                ref, ref_span = match_document(ref_mark, output_ref_span)
                #logger.info(f"For `{target_sent}`, find citations {ref}")
                ref_id = [x -1 for x in ref]
                processed_refs = set()
                ref_passage = []
                for psgs_id in ref_id:
                    if 0 <= psgs_id < len(item['docs']) and psgs_id not in processed_refs:
                        ref_passage.append(_format_document(item['docs'][psgs_id]))
                        processed_refs.add(psgs_id)
                    elif psgs_id in processed_refs:
                        print("Warning: psgs_id already processed:", psgs_id + 1)
                    else:
                        print("Error: psgs_id out of range:", psgs_id+1)

                joint_span = '\n'.join(ref_span)
                joint_passage = '\n'.join(ref_passage)

            elif method=='ALCE':
                ref = list(set([int(r[1:]) for r in re.findall(r"\[\d+", sent)]))
                #logger.info(f"For `{target_sent}`, find citations {ref}")
                ref_id=list(set([int(r[1:])-1 for r in re.findall(r"\[\d+", sent)]))
                processed_refs = set()
                ref_passage = []
                for psgs_id in ref_id:
                    if 0 <= psgs_id < len(item['docs']) and psgs_id not in processed_refs:
                        ref_passage.append(_format_document(item['docs'][psgs_id]))
                        processed_refs.add(psgs_id)
                    elif psgs_id in processed_refs:
                        print("Warning: psgs_id already processed:", psgs_id+1)
                    else:
                        print("Error: psgs_id out of range:", psgs_id+1)
                ref_span=ref_passage
                joint_passage = '\n'.join(ref_passage)
                joint_span=joint_passage


            autoais_log.append({
                "question": item['question'],
                "output_answer": item['output'],
                "docs": item['docs'],
                "claim": {
                    "sentence": sent,
                    "if_citations_needed": need_citations,
                    "has_reference": ref,
                    "doc_recall": None,
                    "quote_recall": None,
                    "granularity_score":None,
                    "granularity_span":None
                }
            })

            if len(ref) == 0:
                # No citations
                joint_entail = 0
                joint_doc_entail=0
            elif any([ref_id > len(item['docs']) for ref_id in ref]):
                # Citations out of range
                joint_entail = 0
                joint_doc_entail=0
            else:
                if at_most_citations is not None:
                    ref = ref[:at_most_citations]
                total_citations += len(ref)

            # 更新正确预测是否需要引用的数量
            if_citations_needed = autoais_log[-1]["claim"]["if_citations_needed"]
            has_reference = autoais_log[-1]["claim"]["has_reference"]
            if (if_citations_needed == 1 and has_reference) or (if_citations_needed == 0 and not has_reference):
                correct_predictions += 1
            #logger.info("citation appropriateness finished")

            # 2. 在需要引用的情况下才计算citation correctness
            if need_citations and has_reference:#需要引用且引用了才考虑后两个指标
                start_time = time.time()
                need_citations_sentences += 1
                # 2.(1):quote_corr
                # If not directly rejected by citation format error, calculate the recall score
                if joint_entail == -1:
                    # φ(premise, hypothesis)判断所有引用span的拼接是否entail模型的回答output
                    joint_entail = _run_nli_autoais(joint_span, target_sent)
                entail += joint_entail
                autoais_log[-1]["claim"]["quote_recall"] = joint_entail
                #logger.info(f"citation recall finished, recall is {joint_entail}")

                #2.(2):doc_corr
                if joint_doc_entail == -1:
                    if method=='ALCE':
                        joint_doc_entail=joint_entail
                    elif method=='baseline':
                        joint_doc_entail=_run_nli_autoais(joint_passage, target_sent)
                entail_doc+=joint_doc_entail
                autoais_log[-1]["claim"]["doc_recall"] = joint_doc_entail
                #print(f"the total time for two recall is {time.time() - start_time}")



                # 4. 只有quote_corr=1（当该条数据，所有引用的拼接可以entail模型output的时候，）才计算引用粒度granularity
                start_time=time.time()
                if joint_entail:
                    all_clauses = []
                    clauses_first_three = []
                    # 遍历每个不同的this_span
                    #logger.info("calculating granularity")
                    if len(ref_span)>5:
                        print("Too many quotations!")
                        autoais_log[-1]["claim"]["granularity_score"] = None
                        autoais_log[-1]["claim"]["granularity_span"] = 0
                    else:
                        for idx, this_span in enumerate(ref_span):
                            #logger.info(f"this span is {this_span}")
                            # 分割引用跨度为子句
                            clauses = re.split(r'([,.])', this_span)
                            clauses = [clause.strip() for clause in clauses if
                                       clause.strip() and any(char.isalnum() for char in clause.strip())]
                            all_clauses.append(clauses)
                            if idx<3:
                                clauses_first_three.append(clauses)

                        max_span_count = len(sum(all_clauses, []))
                        if max_span_count==0:
                            continue
                        doc_count = len(all_clauses)
                        min_comb_length=float('inf')

                        if method=="ALCE" and grain=="default":
                            gold_span_res=run_converge_2(clauses_first_three,target_sent)
                        else:
                            gold_span_res = run_converge_2(all_clauses, target_sent)
                        # gold结果
                        merged_gold_span_res = []

                        # 遍历嵌套列表，并将其中的子列表合并到大列表中
                        for sublist in gold_span_res:
                            merged_gold_span_res.extend(sublist)
                        autoais_log[-1]["claim"]["granularity_span"] = merged_gold_span_res
                        min_comb_length=len(merged_gold_span_res)
                        if min_comb_length!=float('inf'):
                            granularity_score = min_comb_length / max_span_count
                            granularity_list.append(granularity_score)
                            autoais_log[-1]["claim"]["granularity_score"] = granularity_score


                print(autoais_log[-1]["claim"]["granularity_span"])
                print(autoais_log[-1]["claim"]["granularity_score"])
                print(f"the total time for granularity is {time.time() - start_time}")
            else:#不需要引用或没有引用
                autoais_log[-1]['claim']['recall']=None
                autoais_log[-1]["claim"]["granularity_score"]=None
                autoais_log[-1]["claim"]["granularity_span"]=None


        sent_total += len(sents)
        ais_scores_need.append(correct_predictions / len(sents))#是否正确判断需不需要引用：正确判断/总
        if need_citations_sentences!=0:# recall：能entail的/需要引用的
            ais_scores.append(entail / need_citations_sentences)
            ais_doc_scores.append(entail_doc / need_citations_sentences)

        #过滤None
        granularity_list = [value for value in granularity_list if value is not None]

    #logger.info(f"skipped {skipped}")
    #autoais_log.append(f"skipped {skipped}")
    ##print(autoais_log)
    print(ais_scores_need,ais_doc_scores,ais_scores,granularity_list)
    return {
        "citation_correct_prediction": 100 * np.mean(ais_scores_need),
        "citation_doc_rec":100 * np.mean(ais_doc_scores),
        "citation_quote_rec": 100 * np.mean(ais_scores),
        "citation_granularity": 100 * np.mean(granularity_list)
    }#autoais_log

def compute_qampari_f1(data, cot=False):
    prec = []
    rec = []
    rec_top5 = []
    f1 = []
    f1_top5 = []

    num_preds = []
    for item in data:
        if cot:
            if ":" in item['output']:
                o = ':'.join(item['output'].split(":")[1:])  # try to separate the COT part and the answer list part.
            else:
                o = ""
        else:
            o = item['output']
        preds = [normalize_answer(x.strip()) for x in remove_citations(o).rstrip().rstrip(".").rstrip(",").split(",")]
        preds = [p for p in preds if len(p) > 0]  # delete empty answers
        #print(preds)
        num_preds.append(len(preds))
        answers = [[normalize_answer(x) for x in ans] for ans in item['answers']]
        flat_answers = [item for sublist in answers for item in sublist]
        #print(flat_answers)
        prec.append(sum([p in flat_answers for p in preds]) / len(preds) if len(preds) > 0 else 0)
        #print(prec)
        rec.append(sum([any([x in preds for x in a]) for a in answers]) / len(answers))
        rec_top5.append(min(5, sum([any([x in preds for x in a]) for a in answers])) / min(5, len(answers)))
        if (prec[-1] + rec[-1]) == 0:
            f1.append(0)
        else:
            f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]))
        if (prec[-1] + rec_top5[-1]) == 0:
            f1_top5.append(0)
        else:
            f1_top5.append(2 * prec[-1] * rec_top5[-1] / (prec[-1] + rec_top5[-1]))

    return {
        "num_preds": np.mean(num_preds),
        "qampari_prec": 100 * np.mean(prec),
        "qampari_rec": 100 * np.mean(rec),
        "qampari_rec_top5": 100 * np.mean(rec_top5),
        "qampari_f1": 100 * np.mean(f1),
        "qampari_f1_top5": 100 * np.mean(f1_top5),
    }

def compute_length(data):
    return sum(len(item['output'].split(' '))for item in data)/(len(data))
    

if __name__ =='__main__':
    #question = "Why did New York City try to ban food donations to the poor?"
    #output = "New York City, under Mayor Michael Bloomberg's administration, tried to ban food donations to the poor mainly due to concerns about the nutritional content of the donated food. The city argued that it couldn't inspect donated food for its salt, fat, and fiber content, thereby making it hard to control the nutritional quality of the food served to its homeless population [1][2][3]. Critics of this policy, however, have claimed such an approach demonstrated excessive control over people's eating habits and lacked common sense [2]. Despite the ban, many organizations like the New York City Rescue Mission continued to serve needy citizens through food donations [5]."
    #compute_qa(question,output,['',''])
    pass



class Evaluator():
    autoais_model_load = False

    eval_criteria = {'test_pr':test_compute_autoais,'cite_recall_precision':compute_autoais, 'pr':compute_autoais,'qa':compute_qa,'rouge': compute_rouge_l,'claims':compute_claims, 'qampari':compute_qampari_f1,'length':compute_length,'str_em':compute_str_em,'grained':compute_autoais_grained,'cite_recall_precision_llm':lambda data: compute_autoais(data=data,entail_function=_run_llm_autoais),'mauve':compute_mauve}
    def __init__(self,criteria= None, pipeline = None, ais_model = None) -> None:
        self.eval_criteria = Evaluator.eval_criteria
        self.pipeline = pipeline
        self.get_data = {}
        self.ais_model = ais_model
        global ais_LLM
        ais_LLM = ais_model

            

    def set_eval(self, eval_c, **data_get_key):
        if eval_c in self.get_data.keys():
            print(f'Already set! {eval_c}')
            return 
        if eval_c in self.eval_criteria.keys():
            self.get_data[eval_c] = data_get_key
            if eval_c == 'cite_recall_precision':
                global autoais_model, autoais_tokenizer
                if not Evaluator.autoais_model_load:
                    print('Initializing eval model for citation precision and recall...') 
                    autoais_model = AutoModelForSeq2SeqLM.from_pretrained(AUTOAIS_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
                    autoais_tokenizer = AutoTokenizer.from_pretrained(AUTOAIS_MODEL, use_fast=False)
                    Evaluator.autoais_model_load = True
            if eval_c == 'qa':
                global qa_pipeline
                qa_pipeline = transformers.pipeline("question-answering", model=QA_MODEL)

        else:
            raise KeyError('eval_criteria unavailable')
    
    def new_eval(self, name, eval_func, **data_get_key):
        self.eval_criteria[name] = eval_func
        self.set_eval(name, **data_get_key)

    def __call__(self,data_from_pipeline= None):
        result = {}
        
        for criteria, get_data in self.get_data.items():
            if not data_from_pipeline:
                data_dict = {}
                for k, v in get_data.items():
                    if isinstance(v,str):
                        if v == 'output':
                            data_dict[k] = ' '.join(self.pipeline.output)
                        elif v == 'doc_cache':
                            data_dict[k] = self.pipeline.doc_cache
                        else:
                            data_dict[k] = self.pipeline.dataset[self.pipeline.data_index][v]
                    else:
                        data_dict[k] = v
            else:
                data_dict = data_from_pipeline

            eval_func = self.eval_criteria[criteria]
            data = [data_dict]
            result[criteria] = eval_func(data)
        return result
        


class DefaultEvaluator(Evaluator):
    def __init__(self, args = None, criteria= None, pipeline = None) -> None:
        super().__init__(criteria,pipeline)
        if args:
            if  hasattr(args,'pr') and args.pr:
                self.set_eval('cite_recall_precision', output = PIPELINE_OUTPUT, docs = PIPELINE_DOC_CACHE, question = 'question')
            if  hasattr(args,'mauve') and args.mauve:
                self.set_eval('mauve', output = PIPELINE_OUTPUT, answer = 'answer' ,question = 'question')
            if  hasattr(args,'rouge') and args.rouge:
                if (hasattr(args, 'dataset') and 'qampari' not in args.dataset.lower()) or not hasattr(args, 'dataset'):
                    self.set_eval('rouge', output = PIPELINE_OUTPUT, answer = 'answer')
            if  hasattr(args,'qa') and args.qa:
                if (hasattr(args, 'dataset') and 'asqa' in args.dataset.lower()) or not hasattr(args, 'dataset'):
                    self.set_eval('qa',output = PIPELINE_OUTPUT, qa_pairs = 'qa_pairs')
            if  hasattr(args,'claims') and args.claims:
                if (hasattr(args, 'dataset') and 'eli5' in args.dataset.lower()) or not hasattr(args, 'dataset'):
                    self.set_eval('claims',output = PIPELINE_OUTPUT, claims = 'claims')
            if  hasattr(args,'qampari') and args.qampari:
                if (hasattr(args, 'dataset') and 'qampari' in args.dataset.lower()) or not hasattr(args, 'dataset'):
                    self.set_eval('qampari',output = PIPELINE_OUTPUT, answers = 'answers')
            if  hasattr(args,'length') and args.length:
                self.new_eval('length',lambda data: len(data[0]['output'].split(' ')), output = PIPELINE_OUTPUT)

        elif criteria:
            if 'cite_recall_precision' in criteria:
                self.set_eval('cite_recall_precision', output = PIPELINE_OUTPUT, docs = PIPELINE_DOC_CACHE, question = 'question')
            if  hasattr(args,'mauve') and args.mauve:
                self.set_eval('mauve', output = PIPELINE_OUTPUT, answer = 'answer' ,question = 'question')
            if 'rouge' in criteria:
                self.set_eval('rouge', output = PIPELINE_OUTPUT, answer = 'answer')
            if 'qa' in criteria:
                self.set_eval('qa',output = PIPELINE_OUTPUT, qa_pairs = 'qa_pairs')
            if 'str_em' in criteria:
                self.set_eval('str_em',output = PIPELINE_OUTPUT, qa_pairs = 'qa_pairs')
            if 'claims' in criteria:
                self.set_eval('claims',output = PIPELINE_OUTPUT, claims = 'claims')
            if 'qampari' in criteria:
                self.set_eval('qampari',output = PIPELINE_OUTPUT, answers = 'answers')
            if 'length' in criteria:
                self.new_eval('length',lambda data: len(data[0]['output'].split(' ')), output = PIPELINE_OUTPUT)

        else:
            self.new_eval('length',lambda data: len(data[0]['output'].split(' ')), output = PIPELINE_OUTPUT)