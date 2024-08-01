from citekit.cite_modules.LLM import LLM
from citekit.cite_modules.augment_model import Retriever
from citekit.pipeline.pipeline import Pipeline, PIPELINE_OUTPUT,PIPELINE_DOC_CACHE
from citekit.prompt.prompt import Prompt, DocPrompt,ALCEDocPrompt,NewALCEVanillaPrompt
from citekit.Dataset.Dataset import PromptDataset
from citekit.evaluator.evaluator import DefaultEvaluator
from citekit.utils.utils import output_begin_with,output_end_with
import json
import argparse

def one_paragraph(text):
    paras = text.lstrip('\n').split('\n')
    if not paras:
        return ''
    else:
        return paras[0].rstrip('\n')
    
def cut_and_make_as(datakey):
    def f(passage):
        return {datakey:one_paragraph(passage)}
    return f

def if_output(x):
    return not (output_begin_with('check')(x))

def drop_end_and_output(x):
    x = one_paragraph(x)
    if x[-len('End.'):] == 'End.':
        x = x[:-len('End.')]
    if x[:len('output')] =='output' :
        x = x[len('output'):]
    if x[-len('End'):] == 'End':
        x = x[:-len('End')]
    return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='result.json', help="Path to the config file")
    parser.add_argument("--model", type=str, default='gpt-3.5-turbo-0301', help="model name or path")
    parser.add_argument("--shots", type=int, default=2, help="number of shots")
    parser.add_argument("--ndoc", type=int, default=5, help="number of docs")
    parser.add_argument("--pr", type=bool, default=False, help="use cite PR")
    parser.add_argument("--rouge", type=bool, default=False, help="use rouge")
    parser.add_argument("--qa", type=bool, default=False, help="eval qa")
    parser.add_argument("--length", type=bool, default=True, help="eval length")
    parser.add_argument("--temp", type=float, default=0.5, help="temperature")
    parser.add_argument("--claims", type=bool, default=False, help="eval length")
    parser.add_argument("--qampari", type=str, default=False, help="eval qampari")
    parser.add_argument("--turns", type=int, default=6, help="turns")
    parser.add_argument("--dataset", type=str, default='data/asqa_eval_gtr_top100.json', help="dataset")
    parser.add_argument("--demo", type=str, default='prompts/asqa_default.json', help="demo")
    args = parser.parse_args()

    with open('data/asqa_eval_gtr_top100.json','r',encoding='utf-8') as file:
        dataset = json.load(file)
    with open('prompts/asqa_interact_doc_id.json','r',encoding='utf-8') as file:
        demo = json.load(file)
    documents = [DocPrompt().load_data(list(enumerate(data['docs'][:10])),Title = lambda data: data[1]['title'], Passage = lambda data: data[1]['text']) for data in dataset]

    llm_instruction = demo['instruction']
    dataset = PromptDataset(dataset,'question','answer','qa_pairs', extract = lambda data: ''.join(ALCEDocPrompt().default_load_data_extraction(data['docs'][:10])), docs = lambda data: ALCEDocPrompt().default_load_data(data['docs'][:args.ndoc]))[:100]
    shots = '\n'.join(NewALCEVanillaPrompt().load_data(demo['demos'][:args.shots], INST = lambda _:llm_instruction,
                                                  question = lambda data: data['question'], 
                                                  docs = lambda data:''.join(ALCEDocPrompt().default_load_data_extraction(demo['demos'][0]['docs'][:args.ndoc])),
                                                  answer = lambda data: '\n'.join(data['answer'])))
    
    # llm
    llm_prompt = Prompt(template='<shots><INST><question><extract><record><docs><forceAnswer>',components={'INST':'{INST}\n\n', 
                                                                                             'shots':'{shots}\n',
                                                                                        'question':'Question:{question}\n\n',
                                                                                        'extract':'{extract}\n',
                                                                                        'docs':'{docs}',
                                                                                        'record':'Answer:\n{record}',
                                                                                        'forceAnswer':'\n{forceAnswer}'})
    retriever_prompt = Prompt(template='<IDs>',components={'IDs':'{IDs}'})

    llm = LLM(model=args.model, prompt_maker=llm_prompt, self_prompt={'INST':llm_instruction, 'shots': shots+'\n','forceAnswer': 'Answer: \n'},stop=['\n\n'],max_turn=args.turns)

    eval = DefaultEvaluator(args)


    pipeline = Pipeline(llm = llm, head_prompt_maker=llm_prompt,evaluator = eval,dataset = dataset,save_path=args.save_path)
    retriever = Retriever(prompt_maker=retriever_prompt,pipeline=pipeline,topk=3, documents=documents)

    llm.set_target(retriever, output_begin_with('check'), post_processing=cut_and_make_as('IDs'))
    llm.set_target(llm,if_output, post_processing=lambda x: {'forceAnswer': Prompt.UNABLE})
    llm.add_to_head('record')
    llm.set_output(if_output,post_processing= drop_end_and_output, end=False)
    llm.set_output(output_end_with('End.'), post_processing=drop_end_and_output , end = True)
    llm.set_output(output_end_with('End'), post_processing=drop_end_and_output , end = True)

    retriever.set_target(llm ,post_processing=lambda input, output: {'docs': output, 'forceAnswer': 'Output:'})


    pipeline.run_on_dataset(datakeys=['question','extract'],init_docs='docs')


