import json


truncate = lambda x, l:  x[:l]
token_len = len

def combine(*args):
    if all([isinstance(arg,dict) for arg in args]):
        if len(args) == 1: 
            return args[0]
        else:
            combined = args[0].copy()
            combined.update(combine(*args[1:]))
            return combined

default_get = lambda key :  lambda data: data[key]



class Prompt:
    components = {}
    template = ""
    truncate = lambda x, l:  x[:l]
    UNABLE = 'prompt_unable'

    def __init__(self,template='',components={},max_token=8000) -> None:

        '''
        Args:
            template: The way to order and organize each components, use <NAME> to represent a component, <C1><C2>...<Cn>.
            components: The content of a component, use {NAME} to represent the placeholder of corresponding data
            max_token: a list as long as components, representing the max number of tokens for each component, or a int representing the same max_token for all components
        '''

        # template
        self.template = template

        # components
        if isinstance(components,dict):
            for key in components.keys():
                if f'<{str(key)}>' not in self.template:
                    raise Exception('component name not in template!')
            self.components = components

        # max_token
        self.max_token = {}
        if isinstance(max_token,list):
            if len(components)==len(max_token):
                self.max_token = {att:val for (att,val) in zip(components.keys(),max_token)}
            else:
                raise Exception('max_token is not corresponding to components')
        elif isinstance(max_token,int):
            self.max_token_init = max_token
            self.max_token = {att:max_token for att in components}
        else:
            raise TypeError('max_token should be int or list')
        
    def __repr__(self) -> str:
        prompt = self.template
        for key in self.components.keys():
            prompt = prompt.replace(f'<{str(key)}>',self.components[key])
        return prompt
    
    def __str__(self) -> str:
        return repr(self)

    def part_template(self,**kargs):
        '''
        Add components in to the prompt.
        '''
        for part in kargs.keys():
            if f'<{str(part)}>' in self.template:
                self.components[part] = kargs[part]
            else:
                raise Exception('component name not in template!')
    
    def __call__(self, *args,**kargs) -> str:
        return self.make_prompt(*args, **kargs)

    def make_prompt(self,*args,**kargs) -> str:
        '''
        arg: a dictionary containing all contents to the placeholder of the prompt
        kargs: use NAME=value to pass arguments
        '''

        if args:
            args = combine(*args)
            args = args.copy()
            args.update(kargs)
        else:
            args = kargs
        prompt = self.template

        for key in self.components.keys():
            if key not in args or args[key] == Prompt.UNABLE:
                prompt = prompt.replace(f'<{str(key)}>',"")
            else:
                prompt = prompt.replace(f'<{str(key)}>', self.components[key])
        
        prompt_args = {}
        for key in args.keys():
            if key in self.components.keys():
                if self.max_token.get(key):
                    max_token = self.max_token.get(key)
                else:
                    max_token = min(4096,self.max_token_init)
                if token_len(args[key])> max_token:
                    args[key] = self.truncate(args[key],max_token)


            return prompt.format(**args)
    
    def set_max_token(self,**kargs) -> None:
        for key in kargs.keys():
            if key in self.components.keys():
                self.max_token[key] = kargs.get(key)
            else:
                raise KeyError(f'{key} not in Template!')
            
    def load_data(self,data_loader,*keys,**projections):
        '''
        load data to make prompts from a data loader
        projections: the function to get the information from a data.
        '''

        prompts = []
        for data in data_loader:
            l_contents = {key:default_get(key)(data) for key in keys}
            d_contents = {projection:projections[projection](data) for projection in projections.keys()}
            prompts.append(self.make_prompt({**l_contents, **d_contents}))

        return prompts
    




class DocPrompt(Prompt):
    '''
    Containing Doc ID, Title and Passage in order:
    
    Document:[{ID}]
    (Title:{Title}) 
    {Passage}
    '''
    def __init__(self, template='<ID><Title><Passage>', components={'ID':'Document[{ID}]: ','Title':'(Title:{Title})','Passage':'{Passage}\n'}, max_token=4096) -> None:
        super().__init__(template, components, max_token)


class ALCEDocPrompt(Prompt):
    '''
    Containing Doc ID, Title and Passage in order:
    
    Document:[{ID}]
    (Title:{Title}) 
    {Passage}
    '''
    def __init__(self, template='<ID><title><text>', components={'ID':'Document [{ID}]','title':'(Title:{title}): ','text':'{text}\n'}, max_token=4096) -> None:
        super().__init__(template, components, max_token)

    def default_load_data(self,data_loader, text = 'text', from_idx = 0):
        return super().load_data(list(enumerate(data_loader)),text = lambda data: data[1][text],ID = lambda data: str(data[0]+1 + from_idx),title = lambda data: data[1]['title'])

    def default_load_data_wo_ID(self,data_loader):
        return super().load_data(list(enumerate(data_loader)),text = lambda data: data[1]['text'],title = lambda data: data[1]['title'])
    def default_load_data_wo_title(self,data_loader):
        return super().load_data(list(enumerate(data_loader)),text = lambda data: data[1]['text'],ID = lambda data: str(data[0]+1))
    def default_load_data_extraction(self,data_loader):
        return super().load_data(list(enumerate(data_loader)),text = lambda data: data[1]['extraction'],ID = lambda data: str(data[0]+1),title = lambda data: data[1]['title'])
    def default_load_data_summary(self,data_loader):
        return super().load_data(list(enumerate(data_loader)),text = lambda data: data[1]['summary'],ID = lambda data: str(data[0]+1),title = lambda data: data[1]['title'])

class ALCEVanillaPrompt(Prompt):
    '''
    Containing INST(Instruction), Question, Doc and Answer in order:

    {INST}

    Question:{Question}

    {Doc}
    Answer:{Answer}
    '''
    def __init__(self,
                template="<INST><Question><Doc><Answer>\n", 
                components={'INST':'{INST}\n\n', 'Question':'Question:{Question}\n\n','Doc':'{Doc}\n','Answer':'Answer:{Answer}'}, 
                max_token=4096) -> None:
        super().__init__(template, components, max_token)

class NewALCEVanillaPrompt(Prompt):
    '''
    Containing INST(Instruction), Question, Doc and Answer in order:

    {INST}

    Question:{Question}

    {Doc}
    Answer:{Answer}
    '''
    def __init__(self,
                template="<INST><question><docs><answer>\n", 
                components={'INST':'{INST}\n\n', 'question':'Question:{question}\n\n','docs':'{docs}\n','answer':'Answer:{answer}'}, 
                max_token=4096) -> None:
        super().__init__(template, components, max_token)



class AGEEPrompt(Prompt): 
    '''
    Containing INST(Instruction), Question and Doc in order:

    {INST}

    Question:{Question}

    Search Results:{Doc}
    '''
    def __init__(self,
                template="<INST><Question><Doc>\n", 
                components={'INST':'{INST}\n\n', 'Question':'Question:\n{Question}\n','Doc':'Search Results:\n{Doc}\n'}, 
                max_token=4096) -> None:
        super().__init__(template, components, max_token)




alce_prompt= ALCEVanillaPrompt()
#alce_prompt.set_max_token(INST = 10,Doc = 100,Answer = 15)
DocP= DocPrompt()


#print(content['demos'])


#print(data[0])
'''
pps = alce_prompt.load_data(content['demos'],
            INST = lambda _: content['instruction'],
            Question = lambda data: data['question'],
            Doc = lambda data: ''.join(DocPrompt().load_data(list(enumerate(data['docs'])),
                                                              ID = lambda data: str(data[0]),
                                                              Title = lambda data: data[1]['title'],
                                                              Passage = lambda data: data[1]['text'])),
            Answer = lambda data: data['answer'])

'''

#print(pps[0])

'''
data_loader = []
with open('data.txt','r',encoding='utf-8') as f:
    content = f.readlines()
    for i,c in enumerate(content):
        if i%3 == 0:
            data_loader.append({'Q':c.strip(),'A':content[i+1].strip()})
print(data_loader)


pps = Dp.load_data(data_loader,
             INST= lambda data: "Instruction: Write an accurate, engaging, and concise answer for the given question",
             Question= lambda data: data['Q'],
             Answer= lambda data: data['A'])
'''