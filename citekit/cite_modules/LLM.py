import torch
from citekit.prompt.prompt import Prompt
import re
from citekit.utils.utils import one_paragraph, first_sentence
import random
import os

class Module:
    module_count = 1
    def __init__(self,prompt_maker: Prompt = None, pipeline = None, self_prompt = {}, iterative = False, merge = False, max_turn =6, output_as = None, parallel = False) -> None:
        self.self_prompt = self_prompt
        self.use_head_prompt = True
        self.connect_to(pipeline)
        self.prompt_maker = prompt_maker
        self.last_message = ''
        self.destinations = []
        self.conditions = {}
        self.head_key = None
        self.parallel = parallel
        self.iterative = iterative
        self.merge = merge
        self.head_process = one_paragraph
        self.max_turn = max_turn
        self.multi_process = False
        self.output_cond = {} # {cond : {'post_processing':post, 'end':end}}
        self.count = Module.module_count
        Module.module_count += 1
        self.if_add_output_to_head = False

        self.turns = 0
        self.end = False

    def __str__(self) -> str:
        if self.model_type:
            return f'{self.model_type}-[{self.count}]'
        else:
            return f'Unknown-type module-[{self.count}]'
    
    def end_multi(self):
        return

    def set_use_head_prompt(self,use):
        assert isinstance(use,bool)
        self.use_head_prompt = use
    
    def reset(self):
        self.end = False
        self.turns = 0
    
    def change_to_multi_process(self,bool_value):
        if bool_value:
            self.last_message = []
        else:
            self.last_message = ''
        self.multi_process = bool_value
    @property
    def get_use_head_prompt(self):
        return self.use_head_prompt

    def generate(self, head_prompt: dict = {}, dynamic_prompt: dict = {}):
        raise NotImplementedError
    
    def send(self):
        for destination in self.destinations:
            cond = self.conditions[destination]['condition']
            if cond(self):
                return destination
        return None

    def set_target(self,destination, condition = lambda self: True, post_processing = lambda x:x) -> None:
        self.conditions[destination] = {'condition': condition, 'post_processing' : post_processing}
        self.destinations = [destination] + self.destinations
        destination.connect_to(self.pipeline)
    
    def add_output_to_head(self, outputs):
        if self.if_add_output_to_head:
            if not self.head_sub:
                if self.head_key not in self.pipeline.head.keys():
                    self.pipeline.head.update({self.head_key: self.head_process(outputs)})
                else:
                    self.pipeline.head[self.head_key] += '\n'
                    self.pipeline.head[self.head_key] += self.head_process(outputs)
            else: 
                self.pipeline.head[self.head_key] = self.head_process(outputs)

    def connect_to(self, pipeline = None) -> None:
        self.pipeline = pipeline
        if pipeline:
            pipeline.module.append(self)

    def output(self):
        outed = False
        for cond, post_and_end in self.output_cond.items():
            if cond(self):
                if not outed:
                    if not self.merge:
                        self.pipeline.output.append(post_and_end['post_processing'](self.last_message))
                    else:
                        self.pipeline.output.append(post_and_end['post_processing'](''.join(self.last_message)))
                    outed = True
                if post_and_end['end']:
                    self.end =  True

    def set_output(self, cond = lambda self: True, post_processing = lambda x:x, end = True):
        self.output_cond[cond] = {'post_processing': post_processing, 'end' : end}
    
    def get_first_module(self):
        return self
    
    def add_to_head(self, datakey, sub = False, process = None):
        self.if_add_output_to_head = True
        self.head_key = datakey
        self.head_sub = sub
        if process:
            self.head_process = process


def load_model(model_name_or_path,dtype = torch.float16):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map='auto',
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model.eval()
    return model, tokenizer


class LLM(Module):
    def __init__(self, model = None, prompt_maker: Prompt =None, pipeline = None, post_processing = None, self_prompt = {}, device = 'cpu',temperature = 0.5 ,stop = None, max_turn = 6, share_model_with = None, iterative = False, auto_cite = False, output = None,merge = False, noisy = True, parallel = False, output_as ='Answer') -> None:
        super().__init__(prompt_maker,pipeline,self_prompt, iterative, merge, parallel = parallel)
        self.max_turn = max_turn
        if post_processing:
            self.post_processing = post_processing
        else:
            self.post_processing = lambda x: {output_as:x}
        if model:
            self.model_name = model
        self.stop = stop
        self.multi_process = False
        self.noisy = noisy
        self.head_process = one_paragraph
        self.auto_cite = auto_cite
        if auto_cite:
            self.cite_from = 'docs'
        if model:
            if 'gpt' not in model.lower():
                if not share_model_with:
                    print('loading model...')
                    self.model, self.tokenizer = self.load_model(model)
                else: 
                    print('sharing model...')
                    self.model, self.tokenizer = share_model_with.model, share_model_with.tokenizer
                self.temperature = temperature
                self.device = device
            else:
                self.openai_key = os.getenv('OPENAI_API_KEY')
        self.output_cond = {} # {cond : {'post_processing':post, 'end':end}}
        self.if_add_output_to_head = False

        self.token_used = 0

    def reset(self):
        self.end = False
        self.turns = 0
        self.token_used = 0
        
    
    def __str__(self) -> str:
        if self.model_name:
            return f'{self.model_name}-[{self.count}]'
        else:
            return 'unknown model'
    
    def __repr__(self) -> str:
        return (f'{self.prompt_maker}\n|\n|\nV\n{self}\n|\n|\nV\n'+ '/'.join([str(des) for des in self.destinations]+['output']))
    
    def load_model(self, model_name_or_path,dtype = torch.float16):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map='auto',
    )

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model.eval()
        return model, tokenizer

    def set_cite(self,key):
        self.cite_from = key
        self.auto_cite = True
    
    def generate_content(self, prompt):
        if 'gpt' in self.model_name.lower():
            import openai
            openai.api_key = self.openai_key
            prompt = [
                    {'role': 'system',
                     'content': "You are a good helper who follow the instructions"},
                    {'role': 'user', 'content': prompt}
                ]
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=prompt,
                max_tokens=500,
                stop = self.stop
            )
            self.token_used += response['usage']['completion_tokens'] + response['usage']['prompt_tokens']
            return response['choices'][0]['message']['content']
        
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            stop = [] if self.stop is None else self.stop

            outputs = self.model.generate(
                    **inputs,
                    do_sample = True,
                    max_new_tokens = 200,
                    temperature = self.temperature
                    )
            self.token_used += len(outputs[0])

            outputs = self.tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
            return one_paragraph(outputs)
            print(outputs)


    def generate(self, head_prompt: dict = {}, dynamic_prompt: dict = {}):
        if self.use_head_prompt: 
            #print(head_prompt,self.self_prompt,dynamic_prompt)
            prompt = self.prompt_maker(head_prompt,self.self_prompt,dynamic_prompt)
        else:
            prompt = self.prompt_maker(self.self_prompt,dynamic_prompt)
        if self.noisy:
            print(f'prompt to {str(self)}:\n',prompt,'\n\n')
        self.turns += 1
        
        outputs = self.generate_content(prompt)
        #print('DEBUG:',outputs)
        if self.noisy:
            print('OUTPUT:')
            print(outputs)
        if self.auto_cite:
            outputs = self.cite_from_prompt({**head_prompt,**self.self_prompt,**dynamic_prompt},outputs)
        if self.multi_process:
            self.last_message.append(outputs)
        else:
            self.last_message = outputs


        self.add_output_to_head(outputs)

        destination = self.send()

        if self.turns > self.max_turn:
            self.end = True
        if destination in self.conditions:
            return self.conditions[destination]['post_processing'](outputs)
        else: 
            return self.post_processing(outputs)
        
    def add_output_to_head(self, outputs):
        if self.if_add_output_to_head:
            if not self.head_sub:
                if self.head_key not in self.pipeline.head.keys():
                    self.pipeline.head.update({self.head_key: self.head_process(outputs)})
                else:
                    self.pipeline.head[self.head_key] += '\n'
                    self.pipeline.head[self.head_key] += self.head_process(outputs)
            else: 
                self.pipeline.head[self.head_key] = self.head_process(outputs)

    def output(self):
        outed = False
        for cond, post_and_end in self.output_cond.items():
            if cond(self):
                if not outed:
                    if not self.merge and not self.iterative:
                        self.pipeline.output.append(post_and_end['post_processing'](self.last_message))
                    else:
                        self.pipeline.output.append(post_and_end['post_processing'](''.join(self.last_message)))
                    outed = True
                if post_and_end['end']:
                    self.end =  True

    def set_output(self, cond = lambda self: True, post_processing = lambda x:x, end = True):
        self.output_cond[cond] = {'post_processing': post_processing, 'end' : end}

    def cite_from_prompt(self,prompt_dict,input):
        input = first_sentence(input)
        cite_docs = prompt_dict[self.cite_from]
        refs = re.findall(r'\[\d+\]', cite_docs)
        pattern = r'([.!?])\s*$'
        if refs:
            cite = ''.join(refs)
        else:
            cite = ''
        output = re.sub(pattern, rf'{cite}\1 ', input)
        if cite not in output:
            output += cite
        return output
    def add_to_head(self, datakey, sub = False, process = None):
        self.if_add_output_to_head = True
        self.head_key = datakey
        self.head_sub = sub
        if process:
            self.head_process = process



class TestLLM(LLM):
    def __init__(self, model='gpt-4', prompt_maker: Prompt = None, pipeline=None, post_processing=lambda x: x, self_prompt={}, device='cpu', temperature=0.5, stop=None, max_turn=6,share_model_with = None, iterative= False, ans = None) -> None:
        super().__init__(model,prompt_maker,pipeline,self_prompt=self_prompt,share_model_with=share_model_with,iterative=iterative)
        self.max_turn = max_turn
        self.post_processing = post_processing
        self.model_name = model
        self.last_message = ''
        self.stop = stop
        self.output_cond = {} # {cond : {'post_processing':post, 'end':end}}
        self.if_add_output_to_head = False

        self.token_used = 0
        self.ans = 'Strain[1], turns:, heat[2][4]. Sent2[5]. Sent3.\n\n rdd' if not ans else ans
    def generate_content(self, prompt):
        return self.ans


class AutoAISLLM(LLM):
    def __init__(self, model=None, prompt_maker: Prompt = None, pipeline=None, post_processing=None, self_prompt={}, device='cpu', temperature=0.5, stop=None, max_turn=6, share_model_with=None, iterative=False, auto_cite=False, output=None, merge=False, noisy=False, output_as='Answer') -> None:
        super().__init__(model, prompt_maker, pipeline, post_processing, self_prompt, device, temperature, stop, max_turn, share_model_with, iterative, auto_cite, output, merge, noisy, output_as)

        self.prompt_maker = Prompt('<INST><premise><claim>\n Answer: ',components={
            'INST':'{INST}\n\n',
            'premise':'Premise: {premise}\n\n',
            'claim':'Claim: {claim}\n',
        })
        self.self_prompt={'INST': 'In this task, you will be presented a premise and a claim. If the premise entails the claim, output "1", otherwise output "1". Your answer should only contains one number without any other letters and punctuations.'}
    
    def generate(self, premise, claim):
        dict_answer = super().generate({'premise':premise,'claim':claim})
        return dict_answer.get('Answer')



if __name__ == '__main__':
    prompt = Prompt(template='<INST><Question><Docs><feedback><Answer>',components={'INST':'{INST}\n\n', 
                                                                                    'Question':'Question:{Question}\n\n',
                                                                                    'Docs':'{Docs}\n',
                                                                                    'feedback':'Here is the feed back of your last response:{feedback}\n',
                                                                                    'Answer':'Here is answer and you have to give feedback:{Answer}'})
    m = LLM('gpt')