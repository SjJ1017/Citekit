from citekit.cite_modules.LLM import LLM,Module
from citekit.cite_modules.augment_model import Retriever
from citekit.pipeline.pipeline import Pipeline, PIPELINE_OUTPUT, PIPELINE_DOC_CACHE
from citekit.prompt.prompt import Prompt, ALCEVanillaPrompt, DocPrompt,ALCEDocPrompt,NewALCEVanillaPrompt
from citekit.Dataset.Dataset import PromptDataset
from citekit.evaluator.evaluator import Evaluator,DefaultEvaluator
from citekit.utils.utils import output_begin_with, make_as,output_end_with,one_paragraph
import json
import argparse

RECITE_PROMPT = "The answer to the above question can be found in the following Wikipedia page, section, and paragraph:\n"
class R_LLM(LLM):
    def generate_content(self, prompt):
        output=one_paragraph(super().generate_content(prompt))
        self.pipeline.doc_cache.add_doc(doc = output, add_id = True)
        return f'Document [{len(self.pipeline.doc_cache)}]\n'+output
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='result.json', help="Path to the config file")
    parser.add_argument("--recite_model", type=str, default='gpt-3.5-turbo', help="model name or path")
    parser.add_argument("--answer_model", type=str, default='gpt-3.5-turbo', help="model name or path")
    parser.add_argument("--shots", type=int, default=4, help="number of shots")
    parser.add_argument("--ndoc", type=int, default=3, help="number of docs")
    parser.add_argument("--pr", action='store_true', help="use cite PR")
    parser.add_argument("--rouge", action='store_true', help="use rouge")
    parser.add_argument("--temp", type=float, default=0.5, help="temperature")
    parser.add_argument("--qa", action='store_true', help="eval qa")
    parser.add_argument("--mauve",  action='store_true', help="eval mauve")
    parser.add_argument("--length", type=bool, default=True, help="eval claims")
    parser.add_argument("--claims", action='store_true', help="eval length")
    parser.add_argument("--qampari", type=str, default=False, help="eval qampari")
    parser.add_argument("--dataset", type=str, default='data/asqa_eval_gtr_top100.json', help="dataset")
    parser.add_argument("--demo", type=str, default='prompts/asqa_default.json', help="demo")
    args = parser.parse_args()

    file_path = args.dataset
    demo_path = args.demo
    with open(file_path,'r',encoding='utf-8') as file:
        dataset = json.load(file)
    with open(demo_path,'r',encoding='utf-8') as file:
        demo = json.load(file)
    
    llm_inst = demo['instruction']
    prompt = Prompt(template='<passage><shots><question><RP>',components= {'passage':'{passage}\n\n', 'shots':'{shots}\n','question':'Question:{question}\n\n','RP':'\n{RP}\nAnswer:'})
    llm_prompt = Prompt(template='<shots><INST><question><passage>',components= {'INST':'{INST}\n\n','passage':'{passage}\n\nAnswer:\n','question':'Question:{question}\n\n','shots':'{shots}\n'})
    recite_shots = ''.join(NewALCEVanillaPrompt().load_data(demo['demos'][:args.shots],'question', docs = lambda data: RECITE_PROMPT, answer = lambda data: ''.join(ALCEDocPrompt().default_load_data_wo_ID(data['docs'][:1]))))
    llm_shots = '\n'.join(NewALCEVanillaPrompt().load_data(demo['demos'][:args.shots],'question','answer', docs = lambda data: ''.join(ALCEDocPrompt().default_load_data(data['docs'][:args.ndoc]))))
    dataset = PromptDataset(dataset,'question', 'answer', 'answers' ,'qa_pairs','claims')[:1]
    
    # PIPELINE
    eval = DefaultEvaluator(args)
    recite = R_LLM(model=args.recite_model,prompt_maker=prompt, self_prompt={'shots':recite_shots,'RP':RECITE_PROMPT})
    llm = LLM(model=args.answer_model,prompt_maker=llm_prompt, self_prompt={'INST':llm_inst},share_model_with=recite)
    recite.set_target(llm, condition=lambda self:self.turns==args.ndoc,post_processing= lambda x:{'shots':llm_shots})
    recite.add_to_head('passage')
    recite.set_target(recite, condition=lambda self:self.turns<args.ndoc, post_processing=lambda x: {'passage':Prompt.UNABLE})
    pipeline = Pipeline(save_path=args.save_path , llm = llm, module = recite, head_prompt_maker=prompt, evaluator=eval, dataset = dataset)
    llm.set_output(post_processing = one_paragraph ,cond = lambda self: True, end=True)

    # RUN PIPELINE
    pipeline.run_on_dataset(datakeys=['question'],initial_module=recite)
