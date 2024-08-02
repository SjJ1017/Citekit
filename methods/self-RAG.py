from citekit.cite_modules.LLM import LLM
from citekit.cite_modules.augment_model import Retriever,CitationSimplyfier,Verifier,Ranker
from citekit.pipeline.pipeline import Pipeline, PIPELINE_OUTPUT, PIPELINE_DOC_CACHE
from citekit.prompt.prompt import Prompt, ALCEDocPrompt,DocPrompt,NewALCEVanillaPrompt
from citekit.Dataset.Dataset import PromptDataset
from citekit.evaluator.evaluator import DefaultEvaluator, compute_autoais, test_compute_autoais
from citekit.utils.utils import sentence, one_paragraph, each_make_as, each_make_as, make_as,remove_citations, compute_str_em
import json
import argparse

def segment(i,text):
    return [make_as('docs')(doc) for doc in text.split('\n') if doc]



if __name__ == '__main__':

    # SETTING ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='res.json', help="Path to the config file")
    parser.add_argument("--model", type=str, default='gpt-3.5-turbo', help="model name or path")
    parser.add_argument("--shots", type=int, default=2, help="number of shots")
    parser.add_argument("--ndoc", type=int, default=5, help="number of docs")
    parser.add_argument("--pr", action='store_true', help="use cite PR")
    parser.add_argument("--rouge", action='store_true', help="use rouge")
    parser.add_argument("--temp", type=float, default=0.5, help="temperature")
    parser.add_argument("--qa", action='store_true', help="eval qa")
    parser.add_argument("--mauve",  action='store_true', help="eval mauve")
    parser.add_argument("--length", type=bool, default=True, help="eval length")
    parser.add_argument("--claims", action='store_true', help="eval claims")
    parser.add_argument("--qampari", type=str, default=False, help="eval qampari")
    parser.add_argument("--dataset", type=str, default='data/asqa_eval_gtr_top100.json', help="dataset")
    parser.add_argument("--demo", type=str, default='prompts/asqa_default.json', help="demo")
    parser.add_argument("--doctype", type=str, default='text', help="demo")
    parser.add_argument("--data_num", type=int, default=1000, help="num of data")
    parser.add_argument("--mode", type=str, default='text', help="mode-granularity: text, extraction or summary")
    parser.add_argument("--k", type=float, default=1.5, help="coefficient of em")
    parser.add_argument("--topk", type=int, default=2, help="topk")
    args = parser.parse_args()

    def score(data):
        pr = compute_autoais(data)
        p = pr["citation_prec"]
        r = pr["citation_rec"]
        em = compute_str_em(data)
        return p + r + args.k * em
    # DATA LOADING
    file_path = args.dataset
    demo_path = args.demo


    with open(file_path,'r',encoding='utf-8') as file:
        dataset = json.load(file)
    with open(demo_path,'r',encoding='utf-8') as file:
        demo = json.load(file)
    data_num  = min(args.data_num,len(dataset))

    llm_instruction = demo['one_sentence_instruction']
    query_inst = demo["query_instruction"]
    shots = '\n\n'.join(NewALCEVanillaPrompt().load_data([demo['demos'][1],demo['demos'][3]],'question', answer = lambda data: remove_citations(sentence('first')(data['answer'])['first']), INST = lambda _: llm_instruction, docs = lambda data: ''.join(ALCEDocPrompt().default_load_data(data['docs'][1:2]))))


    documents = [DocPrompt().load_data(list(enumerate(data['docs'])),Title = lambda data: data[1]['title'], Passage = lambda data: data[1][args.mode]) for data in dataset]
    
    dataset = PromptDataset(dataset,'question','answer','answers','qa_pairs','claims', docs = lambda data: ALCEDocPrompt().default_load_data(data['docs'][:args.ndoc]))[:data_num]
    
    prompt = Prompt(template='<shots><INST><question><ans><docs>\nAnswer:', components= {'INST':'{INST}\n\n','shots':'{shots}\n','question':'Question:{question}\n\n', 'ans':'Prefix:{ans}\n\n','docs':'{docs}\n'})
    queryprompt = Prompt(template='<INST><question><prev><ans>Please generate one query to help find relevent documents, making sure it is different from previous queries(if provided). your query is:\n',components={'question':'Given the original question: {question}\n','ans':'The context is: {ans}\n','prev':'\nPrevious queries:\n{prev}\n\n','INST':'{INST}\n\n'})

    retriever_prompt = Prompt(template='<query>',components={'query':'{query}'})

    query_generator = LLM(model=args.model, prompt_maker=queryprompt, self_prompt={'INST':query_inst})
    retriever_prompt = Prompt(template='<query>',components={'query':'{query}'})
    eval = DefaultEvaluator(args)
    ranker = Ranker(max_turn=5, iterative= True)
    #ranker.set_eval('length', output = 'answer')
    ranker.new_eval('score', score , output = 'answer', docs = 'doc_cache', qa_pairs = 'qa_pairs')
    # PIPELINE CONSTRUCTING
    llm = LLM(model=args.model,prompt_maker=prompt, self_prompt={'INST':llm_instruction, 'shots':shots}, max_turn= 10, auto_cite=True,share_model_with= query_generator, parallel= True)

    pipeline = Pipeline(save_path=args.save_path, llm = llm ,module=[ranker,query_generator],head_prompt_maker=prompt, evaluator=eval,dataset = dataset)

    retriever = Retriever(prompt_maker=retriever_prompt,pipeline=pipeline, retrieve_by='bm25',documents=documents,topk=args.topk)
    query_generator.set_target(retriever, post_processing=make_as('query'))
    query_generator.add_to_head('prev', sub = False)
    retriever.set_target(llm,post_processing=segment)
    llm.set_target(ranker,post_processing=make_as('answer'))
    ranker.set_output(post_processing=lambda x: x['answer'], end = False)

    ranker.add_to_head('ans',sub = True,  process = lambda text: one_paragraph(text['answer']) )
    ranker.set_target(query_generator,post_processing=lambda x:{'ans': x['answer']})

    # RUN PIPELINE
    pipeline.run_on_dataset(datakeys=['question'],initial_module=query_generator)