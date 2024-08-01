from citekit.cite_modules.LLM import LLM
from citekit.cite_modules.augment_model import Retriever,CitationSimplyfier,Verifier
from citekit.pipeline.pipeline import Pipeline, PIPELINE_OUTPUT, PIPELINE_DOC_CACHE
from citekit.prompt.prompt import Prompt, ALCEDocPrompt,DocPrompt,NewALCEVanillaPrompt
from citekit.Dataset.Dataset import PromptDataset
from citekit.evaluator.evaluator import DefaultEvaluator
from citekit.utils.utils import sentence, one_paragraph, each_make_as, each_make_as, three_sentences
import json
import argparse


if __name__ == '__main__':

    # SETTING ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='res.json', help="Path to the config file")
    parser.add_argument("--model", type=str, default='gpt-3.5-turbo', help="model name or path")
    parser.add_argument("--shots", type=int, default=2, help="number of shots")
    parser.add_argument("--ndoc", type=int, default=5, help="number of docs")
    parser.add_argument("--pr", type=bool, default=False, help="use cite PR")
    parser.add_argument("--rouge", type=bool, default=False, help="use rouge")
    parser.add_argument("--temp", type=float, default=0.5, help="temperature")
    parser.add_argument("--qa", type=bool, default=False, help="eval qa")
    parser.add_argument("--length", type=bool, default=True, help="eval length")
    parser.add_argument("--claims", type=bool, default=False, help="eval length")
    parser.add_argument("--qampari", type=str, default=False, help="eval qampari")
    parser.add_argument("--dataset", type=str, default='data/asqa_eval_gtr_top100.json', help="dataset")
    parser.add_argument("--demo", type=str, default='prompts/asqa_default.json', help="demo")
    parser.add_argument("--doctype", type=str, default='text', help="demo")
    parser.add_argument("--mode", type=str, default='vanilla', help="mode")
    parser.add_argument("--data_num", type=int, default=200, help="num of data")
    args = parser.parse_args()

    # DATA LOADING
    file_path = args.dataset
    demo_path = args.demo


    with open(file_path,'r',encoding='utf-8') as file:
        dataset = json.load(file)
    with open(demo_path,'r',encoding='utf-8') as file:
        demo = json.load(file)
    data_num  = min(args.data_num,len(dataset))
    
    llm_instruction = demo['instruction']
    shots = '\n\n'.join(NewALCEVanillaPrompt().load_data([demo['demos'][1],demo['demos'][3]],'question','answer', INST = lambda _: llm_instruction, docs = lambda data: ''.join(ALCEDocPrompt().default_load_data(data['docs'][:args.ndoc]))))
    documents = [DocPrompt().load_data(list(enumerate(data['docs'])),Title = lambda data: data[1]['title'], Passage = lambda data: data[1]['text']) for data in dataset]
    
    if args.doctype == 'text':
        dataset = PromptDataset(dataset,'question','answer','answers','qa_pairs','claims', docs = lambda data: ALCEDocPrompt().default_load_data(data['docs'][:args.ndoc]))[:data_num]
    elif args.doctype == 'summary':
        dataset = PromptDataset(dataset,'question','answer','answers','qa_pairs','claims', docs = lambda data: ALCEDocPrompt().default_load_data_summary(data['docs'][:args.ndoc]))[:data_num]
    elif args.doctype == 'extraction':
        dataset = PromptDataset(dataset,'question','answer','answers','qa_pairs','claims', docs = lambda data: ALCEDocPrompt().default_load_data_extraction(data['docs'][:args.ndoc]))[:data_num]
    
    prompt = Prompt(template='<shots><INST><question><docs>\nAnswer: \n', components= {'INST':'{INST}\n\n','shots':'{shots}\n','question':'Question:{question}\n\n', 'docs':'{docs}\n'})
    queryprompt = Prompt(template='<q><answer><qg_num>',components={'q':'Given the original question: {q}\n','answer':'The claim is: {answer}\n','qg_num':'Please generate up to {qg_num} questions that can help verify the claim with the following constraints: \n1. You should output no more than {qg_num} questions. \n2. The generated questions should be diverse and focus on different aspects of the given claim. \nGenerated questions:'})
    retriever_prompt = Prompt(template='<query>',components={'query':'{query}'})
    eval = DefaultEvaluator(args)

    # PIPELINE CONSTRUCTING
    llm = LLM(model=args.model,prompt_maker=prompt, self_prompt={'INST':llm_instruction, 'shots':shots}, max_turn = 3)
    regen_llm = LLM(model=args.model,prompt_maker=prompt, self_prompt={'INST':llm_instruction, 'shots':shots}, max_turn = 3,share_model_with=llm)
    simplifier = CitationSimplyfier()
    verifier = Verifier()
    query_generator = LLM(model=args.model,prompt_maker=queryprompt, self_prompt={'qg_num':'2'})

    

    pipeline = Pipeline(save_path=args.save_path , llm = llm, module = [simplifier,verifier,query_generator],head_prompt_maker=prompt, evaluator=eval,dataset = dataset,train_data=True)
    retriever = Retriever(prompt_maker=retriever_prompt,pipeline=pipeline, retrieve_by='bm25',documents=documents, topk=1, merge = True)
    if args.mode == 'vanilla':
        llm.set_output(post_processing = one_paragraph, cond = lambda self: True, end=True)
    elif args.mode == 'simplify':
        llm.set_target(simplifier, post_processing = each_make_as('answer'))
        simplifier.set_output()
    elif args.mode == 'VTG':
        llm.set_target(verifier, post_processing = three_sentences('answer'))
        verifier.set_target(simplifier, condition = lambda self: self.last_message or self.turns == 3)
        verifier.set_target(query_generator, condition = lambda self: not self.last_message)
        query_generator.set_target(retriever,post_processing=each_make_as('query'))
        retriever.set_target(regen_llm, post_processing = lambda i,o: {'docs': o})
        regen_llm.set_target(verifier, post_processing = sentence('answer'))
        simplifier.set_output()

    # RUN PIPELINE
    pipeline.run_on_dataset(datakeys=['question','docs'], init_docs='docs')