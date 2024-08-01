from citekit.cite_modules.LLM import LLM
from citekit.pipeline.pipeline import Pipeline, PIPELINE_OUTPUT,PIPELINE_DOC_CACHE
from citekit.prompt.prompt import Prompt, ALCEDocPrompt
from citekit.Dataset.Dataset import PromptDataset
from citekit.evaluator.evaluator import DefaultEvaluator
import argparse
import json
from citekit.utils.utils import cut_and_make_as,one_paragraph,make_as

PARA_SEP = '\n\n'

class FuncLLM(LLM):
    def generate(self, head_prompt: dict = {}, dynamic_prompt: dict = {}):
        docs = head_prompt['docs']
        ans_docs = one_paragraph(dynamic_prompt['span']).split("\n")
        spans = [ans_doc[14:].split("<SPAN_DELIM>") for ans_doc in ans_docs]
        msg = ''
        span_list = {}
        doc_map = {}
        j = 1
        i = 1
        for doc in spans:
            if doc!= [] :
                span_list[f'{i}'] = []
                msg += f'Document [{i}]:\n'
                for span in doc:
                    if len(span)> 3:
                        msg += f'{j}. {span}\n'
                        span_list[f'{i}'].append(f'{j}. {span.strip()}')
                        doc_map[str(j)] = str(i)
                        j+=1
                        docs = docs.replace(span.strip(), f'<highlight_start>{span.strip()}<highlight_end>')
                i+=1
        self.pipeline.head['doc_map'] = doc_map
        self.pipeline.head['docs'] = docs
        self.pipeline.head['span'] = msg
        self.pipeline.head['span_list'] = span_list
        return {'span_list': Prompt.UNABLE,'doc_map': Prompt.UNABLE}
    
class FuncLLM2(LLM):
    def generate(self, head_prompt: dict = {}, dynamic_prompt: dict = {}):
        span_ls = head_prompt['span_list']
        doc_map = head_prompt['doc_map']
        span_list = [item for sublist in head_prompt['span_list'].values() for item in sublist]
        clusters = eval(one_paragraph(dynamic_prompt['cls'].strip()))
        self.pipeline.head['clusters'] = clusters
        def _form(cls):
            text = ''
            doc_list = cls['cluster']
            for doc_num in span_ls.keys():
                pieces = [str(i) for i in doc_list if doc_map.get(str(i),'None') == doc_num]
                if pieces:
                    text += f'Document [{doc_num}]: \n' + '\n'.join([span_list[int(num)-1] for num in pieces])  + '\n'

            return(text)
        #print([{'span': _form(cls)} for cls in clusters])
        return [{'span': _form(cls),'span_list': Prompt.UNABLE,'doc_map': Prompt.UNABLE,'clusters':Prompt.UNABLE} for cls in clusters if _form(cls)]



if __name__ == '__main__':

    # SETTING ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='res.json', help="Path to the config file")
    parser.add_argument("--model", type=str, default='gpt-3.5-turbo', help="model name or path")
    parser.add_argument("--shots", type=int, default=1, help="number of shots")
    parser.add_argument("--ndoc", type=int, default=2, help="number of docs")
    parser.add_argument("--pr", type=bool, default=False, help="use cite PR")
    parser.add_argument("--rouge", type=bool, default=False, help="use rouge")
    parser.add_argument("--temp", type=float, default=0.5, help="temperature")
    parser.add_argument("--qa", type=bool, default=False, help="eval qa")
    parser.add_argument("--length", type=bool, default=True, help="eval length")
    parser.add_argument("--claims", type=bool, default=False, help="eval length")
    parser.add_argument("--qampari", type=str, default=False, help="eval qampari")
    parser.add_argument("--turns", type=int, default=1, help="k")
    parser.add_argument("--use_fast_pr", type=str, default=False, help="test")
    parser.add_argument("--dataset", type=str, default='data/asqa_eval_gtr_top100.json', help="dataset")
    parser.add_argument("--demo", type=str, default='prompts/AnG.json', help="demo")
    args = parser.parse_args()

    # DATA LOADING
    file_path = args.dataset
    demo_path = args.demo
    with open(file_path,'r',encoding='utf-8') as file:
        dataset = json.load(file)
    with open(demo_path,'r',encoding='utf-8') as file:
        demo = json.load(file)[1]

    dataset =PromptDataset(dataset,'question','answer','answers','qa_pairs','claims', docs = lambda data: ALCEDocPrompt().default_load_data_wo_title(data['docs'][:args.ndoc]))[:200]
    selection_shot = demo['selection_instruction'] + PARA_SEP + demo['selection_shot'] + PARA_SEP
    cls_shot = demo['clustering_instruction'] + PARA_SEP + demo['clustering_shot'] + PARA_SEP
    gen_shot = demo['gen_instruction'] + PARA_SEP + demo['gen_shot'] + PARA_SEP

    prompt = Prompt(template='<shot><INST><question><docs><prefix><span><add>',
                    components={'INST':'{INST}\n\n', 
                                'shot':'{shot}',
                                'question':'Question:{question}\n\n',
                                'docs':'{docs}\n',
                                'span':'The highlighted spans are: \n{span}\n\n',
                                'prefix':'Prefix: {prefix}\n\n',
                                'add':'Answer: \n{add}'
                                })

    # PIPELINE
    evaluator = DefaultEvaluator(args)

    select = LLM(model = args.model, prompt_maker = prompt, self_prompt={'INST':demo['selection_instruction'],'shot':selection_shot,'add':''})
    post_select = FuncLLM()
    clustering =  LLM(model = args.model, prompt_maker = prompt, self_prompt={'INST':demo['clustering_instruction'],'shot':cls_shot, 'add':'The highlighted spans are clustered as follows:'},share_model_with=select)
    post_cls = FuncLLM2()
    answer = LLM(model = args.model, prompt_maker = prompt, self_prompt={'INST':demo['gen_instruction'],'shot':gen_shot, 'add':'The next sentence is:'}, share_model_with=select,post_processing=cut_and_make_as('prefix'),iterative=True)

    select.set_target(post_select, post_processing=make_as('span'))
    post_select.set_target(clustering)
    clustering.set_target(post_cls,post_processing=make_as('cls'))
    post_cls.set_target(answer)

    pipeline = Pipeline(save_path=args.save_path, llm=select, module= [post_select,clustering,post_cls,answer], evaluator = evaluator, dataset=dataset)

    answer.set_output(post_processing=lambda ls: ''.join(map(one_paragraph,ls)))
    print(1)
    pipeline.run_on_dataset(datakeys=['question','docs'],init_docs='docs')
