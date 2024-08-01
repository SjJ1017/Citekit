from citekit.cite_modules.LLM import LLM,AutoAISLLM
from citekit.cite_modules.augment_model import AttributingModule
from citekit.pipeline.pipeline import Sequence
from citekit.prompt.prompt import Prompt
from citekit.Dataset.Dataset import FileDataset
from citekit.evaluator.evaluator import DefaultEvaluator,Evaluator,PIPELINE_DOC_CACHE,PIPELINE_OUTPUT,_run_llm_autoais
from citekit.utils.utils import make_as
import json

# DATA LOADING
dataset = FileDataset('data/asqa.json')

prompt = Prompt(template='<INST><question><docs><prefix><span>Answer: ',
                components={'INST':'{INST}\n\n', 
                            'question':'Question:{question}\n\n',
                            'docs':'{docs}\n',
                            'span':'The highlighted spans are: \n{span}\n\n',
                            'prefix':'Prefix: {prefix}\n\n',
                            })


with open('prompts/asqa.json','r',encoding='utf-8') as file:
        demo = json.load(file)
        instruction =  demo['instruction_1']


ais = AutoAISLLM(model = 'gpt-3.5-turbo')
eval = Evaluator(ais_model=ais)
eval.set_eval('cite_recall_precision_llm', output = PIPELINE_OUTPUT, docs = PIPELINE_DOC_CACHE, question = 'question',entail_function = _run_llm_autoais)

llm = LLM(model='gpt-3.5-turbo',prompt_maker=prompt, self_prompt={'INST':instruction})


pipeline = Sequence(sequence=[llm], head_prompt_maker=prompt, evaluator=eval, dataset = dataset)


# RUN PIPELINE
pipeline.run_on_dataset(datakeys=['question','docs'], init_docs='docs')