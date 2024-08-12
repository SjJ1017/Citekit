from citekit.cite_modules.LLM import LLM
from citekit.cite_modules.augment_model import Retriever
from citekit.pipeline.pipeline import Pipeline, PIPELINE_OUTPUT,PIPELINE_DOC_CACHE
from citekit.prompt.prompt import Prompt, DocPrompt
from citekit.Dataset.Dataset import PromptDataset
from citekit.evaluator.evaluator import DefaultEvaluator
from citekit.utils.utils import output_begin_with, make_as,output_end_with,one_paragraph,remove_citations
import json
import argparse
import nltk
import re

def each_make_as(key):
    def function(output):
        sents = nltk.sent_tokenize(one_paragraph(output))
        if len(sents)>3:
            sents = sents[:3]
        return [make_as(key)(sent) for sent in sents]
    return function

def add_citation(ls):
    output = ''
    pattern = r'([.!?])\s*$'
    for i, answer in enumerate(ls):
        cite = f'[{i+1}]'
        answer = one_paragraph(answer)
        if not answer:
            return cite
        else:
            answer = re.sub(pattern, rf'{cite}\1 ', answer)
            if cite not in answer:
                answer += cite
        output += answer
    return output

if __name__ == '__main__':
    # SETTING ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='res.json', help="Path to the config file")
    parser.add_argument("--model", type=str, default='gpt-3.5-turbo', help="model name or path")
    parser.add_argument("--shots", type=int, default=2, help="number of shots")
    parser.add_argument("--ndoc", type=int, default=3, help="number of docs")
    parser.add_argument("--pr", action='store_true', help="use cite PR")
    parser.add_argument("--rouge", action='store_true', help="use rouge")
    parser.add_argument("--temp", type=float, default=0.5, help="temperature")
    parser.add_argument("--qa", action='store_true', help="eval qa")
    parser.add_argument("--mauve",  action='store_true', help="eval mauve")
    parser.add_argument("--length", type=bool, default=True, help="eval length")
    parser.add_argument("--claims", action='store_true', help="eval length")
    parser.add_argument("--qampari", type=str, default=False, help="eval qampari")
    parser.add_argument("--dataset", type=str, default='data/asqa_eval_gtr_top100.json', help="dataset")
    parser.add_argument("--demo", type=str, default='prompts/asqa_default.json', help="demo")
    parser.add_argument("--add_cite", action='store_true', help="manuel add cite")
    parser.add_argument("--top_k", type=int, default=1, help="retrieve docs")
    args = parser.parse_args()

    # DATA LOADING
    file_path = args.dataset
    demo_path = args.demo
    with open(file_path,'r',encoding='utf-8') as file:
        dataset = json.load(file)
    with open(demo_path,'r',encoding='utf-8') as file:
        demo = json.load(file)

    # DATA
    documents = [DocPrompt().load_data(list(enumerate(data['docs'])),Title = lambda data: data[1]['title'], Passage = lambda data: data[1]['text']) for data in dataset]

    dataset =PromptDataset(dataset, 'question','answer','qa_pairs','answers','claims')[:200]
    
    llm_instruction = 'Instruction: Write an accurate, engaging, and concise answer for the given question. Use an unbiased and journalistic tone.'
    if args.add_cite:
        llm_instruction_after = 'Instruction: Revise and correct the answer to an accurate, engaging, and concise answer for the given question using only the provided document using only one sentence. Use an unbiased and journalistic tone. Your revised answer must contain only one short sentence.'
    else:
        llm_instruction_after = 'Instruction: Revise and correct the answer to an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. Use an unbiased and journalistic tone. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents. Your revised answer must contain only one short sentence.'
    shots = '\n\n'.join(llm_instruction + '\n\nQuestion: '+ d['question']+'\n\nAnswer: '+remove_citations(d['answer']) for d in demo['demos'][:args.shots])
    llm_prompt = Prompt(template='<shots><INST><question><docs><answer>\n\nAnswer: ',components={'INST':'{INST}\n\n', 'shots':'{shots}\n\n', 'question':'Question: {question}\n\n','docs':'{docs}', 'answer':'\nThis is the answer you should revise based on the provided document: \n{answer}'})
    retriever_prompt = Prompt(template='<query>',components={'query':'{query}'})
    
    # PIPELINE 
    llm = LLM(model=args.model, prompt_maker=llm_prompt, self_prompt={'INST':llm_instruction,'shots':shots},stop=['\n','\n\n'])
    eval = DefaultEvaluator(args)
    pipeline = Pipeline(llm = llm, head_prompt_maker=llm_prompt,evaluator = eval,dataset = dataset,save_path=args.save_path)
    retriever = Retriever(prompt_maker=retriever_prompt,pipeline=pipeline,retrieve_by='bm25',documents=documents,topk=args.top_k)
    llm.set_target(retriever,lambda self: self.turns == 1, post_processing=each_make_as('query'))
    if args.add_cite:
        llm.set_output(lambda self: self.turns > 1, post_processing = add_citation, end=True)
    else:
        llm.set_output(lambda self: self.turns > 1, post_processing = lambda ls: ''.join(map(one_paragraph,ls)), end=True)
    retriever.set_target(llm ,post_processing=lambda input, output: {'docs': output,'answer': input,'INST':llm_instruction_after,'shots':Prompt.UNABLE})

    # RUN PIPELINE
    pipeline.run_on_dataset(datakeys=['question'])


