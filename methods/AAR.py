from citekit.cite_modules.LLM import LLM
from citekit.cite_modules.augment_model import EvalModule
from citekit.pipeline.pipeline import Pipeline, PIPELINE_OUTPUT,PIPELINE_DOC_CACHE
from citekit.prompt.prompt import Prompt, ALCEVanillaPrompt, DocPrompt,ALCEDocPrompt,NewALCEVanillaPrompt
from citekit.Dataset.Dataset import PromptDataset
from citekit.evaluator.evaluator import Evaluator,DefaultEvaluator
import argparse
import json
from citekit.utils.utils import cut_and_make_as,one_paragraph,make_as

test = False
def convert_result(result):
    key = 'cite_recall_precision'
    if test:
        key = 'test_pr'
    p = result[key]['citation_prec']
    r = result[key]['citation_rec']
    text = f'citation_precision_score: {p}\ncitation_recall_score: {r}\n'
    return {'score':text, 'shots':Prompt.UNABLE}

if __name__ == '__main__':

    # SETTING ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default='res.json', help="Path to the config file")
    parser.add_argument("--model", type=str, default='gpt-3.5-turbo', help="model name or path")
    parser.add_argument("--shots", type=int, default=1, help="number of shots")
    parser.add_argument("--ndoc", type=int, default=3, help="number of docs")
    parser.add_argument("--pr", action='store_true', help="use cite PR")
    parser.add_argument("--rouge", action='store_true', help="use rouge")
    parser.add_argument("--temp", type=float, default=0.5, help="temperature")
    parser.add_argument("--qa", action='store_true', help="eval qa")
    parser.add_argument("--mauve",  action='store_true', help="eval mauve")
    parser.add_argument("--length", type=bool, default=True, help="eval length")
    parser.add_argument("--claims", action='store_true', help="eval claims")
    parser.add_argument("--qampari", type=str, default=False, help="eval qampari")
    parser.add_argument("--turns", type=int, default=1, help="k")
    parser.add_argument("--use_fast_pr", type=str, default=False, help="test")
    parser.add_argument("--dataset", type=str, default='data/asqa_eval_gtr_top100.json', help="dataset")
    parser.add_argument("--demo", type=str, default='prompts/asqa_default.json', help="demo")
    args = parser.parse_args()

    # DATA LOADING
    file_path = args.dataset
    demo_path = args.demo
    with open(file_path,'r',encoding='utf-8') as file:
        dataset = json.load(file)
    with open(demo_path,'r',encoding='utf-8') as file:
        demo = json.load(file)

    dataset =PromptDataset(dataset,'question','answer','answers','qa_pairs','claims', docs = lambda data: ALCEDocPrompt().default_load_data(data['docs'][:args.ndoc]))[:200]
    revise_instruction = '''You will be provided with a set of questions, search results, and corresponding answers. Your task is to evaluate each answer and provide feedback to enhance its quality. Following <Feedback Instruction>, offer specific feedback according to the reward scores for the following aspects: Citation Recall, and Citation Precision.\n\n<Feedback Instruction> \n\n1) Citation Recall: If the reward score is below {citation_recall_score}, provide feedback to offer citations from credible sources for each factual statement you make. If the score is above {citation_recall_score}, affirm that performance on citation recall is satisfactory. \n\n2) Citation Precision: If the reward score is below {citation_score}, provide feedback to cite properly, ensuring all factual statements refer to an appropriate search result. If the score is above {citation_precision_score}, affirm that performance on citation precision is satisfactory.'''
    llm_instruction = demo['instruction']
    re_ans_instruction = '''You will be provided with a set of questions, search results, corresponding answers, the score of answer and the feedback. Your task is to refine the answer, only when the citation score is low. Please still generate an answer with accurate citation.'''
    
    shots = '\n'.join(NewALCEVanillaPrompt().load_data(demo['demos'][:args.shots],'question','answer',INST = lambda _: llm_instruction, docs = lambda data: ''.join(ALCEDocPrompt().default_load_data(data['docs'][:args.ndoc]))))
    prompt = Prompt(template='<shots><INST><question><docs><previousAnswer><score><feedback><answer>\n\nAnswer:',
                    components={'INST':'{INST}\n\n', 
                                'question':'Question:{question}\n\n',
                                'docs':'{docs}\n',
                                'shots':'{shots}\n\n',
                                'score':'Here is the score of the initial answer:\n{score}\n', 
                                'previousAnswer':'Initial Answer:\n{previousAnswer}\n\n',
                                'feedback':'feedback:\n{feedback}\n',
                                'answer':'Here is answer and you have to give feedback on:\n{answer}'})


    # PIPELINE
    llm = LLM(model=args.model, prompt_maker=prompt, self_prompt={'INST':llm_instruction,'shots':shots})
    revise = LLM(model=args.model, prompt_maker=prompt, self_prompt={'INST':revise_instruction,'shots':Prompt.UNABLE}, share_model_with=llm)
    re_ans = LLM(model=args.model, prompt_maker=prompt, self_prompt={'INST':re_ans_instruction}, share_model_with=llm)
    auto_eval_model = EvalModule()
    eval = DefaultEvaluator(args)

    if args.use_fast_pr:
        auto_eval_model.set_eval('test_pr', output = 'previousAnswer', docs = PIPELINE_DOC_CACHE) 
        test = True 
    else:
        auto_eval_model.set_eval('cite_recall_precision', output = 'previousAnswer', docs = PIPELINE_DOC_CACHE, question = 'question')
    pipeline = Pipeline(save_path=args.save_path, llm = llm, module = [auto_eval_model,revise,re_ans],head_prompt_maker=prompt,evaluator = eval,dataset = dataset)
    llm.set_target(auto_eval_model,lambda self: self.turns < args.turns + 1 , post_processing=cut_and_make_as('previousAnswer'))
    re_ans.set_output(lambda self: True, post_processing=one_paragraph, end=True)
    auto_eval_model.set_target(revise, lambda self: self.turns < args.turns + 1, post_processing = convert_result)
    revise.set_target(re_ans, post_processing=make_as('feedback'))
    llm.add_to_head('previousAnswer', sub = True)

    pipeline.run_on_dataset(datakeys=['question','docs'],init_docs='docs')



