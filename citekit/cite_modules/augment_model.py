from citekit.cite_modules.LLM import Module,LLM
from citekit.cite_modules.Retrieve import DPRRetriever
from citekit.evaluator.evaluator import _run_nli_autoais, Evaluator
from citekit.prompt.prompt import Prompt
from citekit.utils.utils import one_paragraph, make_as
from sentence_transformers import SentenceTransformer
import re
import random

    
class Retriever(Module):
    model_type = 'retriever'
    def __init__(self, documents = None ,retrieve_by = 'index', prompt_maker = None, pipeline = None, post_processing = lambda input, output: {'RetrievedDocs':output}, self_prompt = {},topk = 3,add_id = True, merge = False, tsv_path = 'None', emb_path = 'None') -> None:
        super().__init__(prompt_maker,pipeline,self_prompt,merge=merge)
        self.retrieve_by = retrieve_by
        self.use_head_prompt = False
        if not documents:
            self.documents = self.pipeline.doc_cache
        else:
            self.dataset_documents = documents
        self.post_processing = post_processing
        self.add_output_to_head = False
        self.topk = topk
        self.add_id = add_id
        if retrieve_by =='bm25':
            self.bm25_module_loaded = False
            from rank_bm25 import BM25Okapi
            import nltk
            nltk.download('punkt')
            from nltk.tokenize import word_tokenize
            self.word_tokenize = word_tokenize
            self.BM25Okapi = BM25Okapi
            self.bm25_module_loaded = True
        elif retrieve_by == 'dpr':
            self.dpr_retriever = DPRRetriever(DPR_WIKI_TSV=tsv_path, 
                        GTR_EMB=emb_path)
    
    def generate(self,head_prompt: dict = {}, dynamic_prompt: dict = {}):
        index = self.pipeline.data_index
        if isinstance(self.dataset_documents[0], list):
            # Each data has a document
            self.documents = self.dataset_documents[index]
        else:
            self.documents = self.dataset_documents
        # query
        if self.use_head_prompt: 
            prompt = self.prompt_maker(head_prompt,self.self_prompt,dynamic_prompt)
        else:
            prompt = self.prompt_maker(self.self_prompt,dynamic_prompt)   

        retrieved_docs = []
        if self.retrieve_by == 'index':
            # query : Document [1][2]
            indice = [int(r[1:]) - 1 for r in re.findall(r"\[\d+",prompt)]
            for index in indice:
                retrieved_docs.append(self.documents[index])
            if len(retrieved_docs) > self.topk:
                retrieved_docs = retrieved_docs[:self.topk]
        elif self.retrieve_by =='bm25':
                
            # natural language query
            tokenized_docs = [self.word_tokenize(doc.lower()) for doc in self.documents]
            bm25 = self.BM25Okapi(tokenized_docs)
            tokenized_query = self.word_tokenize(prompt.lower()) 
            doc_scores = bm25.get_scores(tokenized_query)
            if self.topk > len(doc_scores):
                self.topk = len(doc_scores) - 1
            top_k_idx = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:self.topk]
            retrieved_docs = [self.documents[index] for index in top_k_idx]
            retrieved_docs_new = []
            for re_doc in retrieved_docs:
                self.pipeline.doc_cache.add_doc(re_doc,self.add_id)
                retrieved_docs_new.append(self.pipeline.doc_cache.get_last())
            retrieved_docs = retrieved_docs_new
            #raise NotImplementedError
        

        elif self.retrieve_by =='gtr':
            docs_dict  = self.dpr_retriever.retrieve(prompt,topk=self.topk)
            retrieved_docs = [f"({d['title']}) {d['text']}" for d in docs_dict]
            retrieved_docs_new = []
            for re_doc in retrieved_docs:
                self.pipeline.doc_cache.add_doc(re_doc,self.add_id)
                retrieved_docs_new.append(self.pipeline.doc_cache.get_last())
            retrieved_docs = retrieved_docs_new
        
        retrieved_docs_prompt = '\n'.join(retrieved_docs)
        destination = self.send()
        if self.multi_process:
            self.last_message.append(retrieved_docs_prompt)
        else:
            self.last_message = retrieved_docs_prompt
        #print(self.last_message)

        if self.add_output_to_head:
            self.pipeline.head.update({self.head_key:retrieved_docs_prompt})
        if destination in self.conditions:
            return self.conditions[destination]['post_processing'](prompt,retrieved_docs_prompt)
        else:
            return retrieved_docs_prompt
        raise NotImplementedError


class EvalModule(Module, Evaluator):
    model_type = 'evaluator'
    def __init__(self, prompt_maker = None, pipeline=None, self_prompt={},criteria = None, iterative = False, max_turn =6 ,parallel = False) -> None:
        Module.__init__(self,prompt_maker, pipeline, self_prompt,iterative=iterative,max_turn=max_turn, parallel=parallel)
        Evaluator.__init__(self,criteria,pipeline)
    
    def generate(self, head_prompt: dict = {}, dynamic_prompt: dict = {}):
        result = {}
        p_data = {**head_prompt, **self.self_prompt,**dynamic_prompt}
        for criteria, get_data in self.get_data.items():
            data_dict = {}
            for k, v in get_data.items():
                if v == 'doc_cache':
                    data_dict[k] = self.pipeline.doc_cache.show_docs()
                else:
                    if v in p_data.keys():
                        data_dict[k] = p_data[v]
                    else:
                        data_dict[k] = self.pipeline.current_data[v]
            eval_func = Evaluator.eval_criteria[criteria]
            data = [data_dict]
            result[criteria] = eval_func(data)

        if self.multi_process:
            self.last_message.append(result)
        else:
            self.last_message = result
        destination = self.send()
        if destination in self.conditions:
            return self.conditions[destination]['post_processing'](result)
        else:
            return result


class CitationSimplyfier(Module):
    '''
    Simplify the citation of the 'answer' in prompt.  
    Argument can be changed to fit into different name of key in prompts 
    By Defaut, the simplifier simplifies the 'answer' and output the sring with citation simplified.
    '''
    model_type = 'simplifier' 
    def __init__(self, prompt_maker = None, pipeline=None, self_prompt={}, criteria = None, key = 'answer', test = False) -> None:
        Module.__init__(self,prompt_maker, pipeline, self_prompt)
        if not test: 
            self.entail = _run_nli_autoais
        else:
            self.entail = lambda p,c : random.randint(0,1)
            self.docs = ['0'] * 100
        self.key = key

    def generate(self, head_prompt: dict = {}, dynamic_prompt: dict = {}) -> str:
        docs = self.pipeline.doc_cache
        prompt = {**head_prompt, **dynamic_prompt}
        answer = prompt[self.key]

        refs = re.findall(r'\[\d+\]', answer)
        last_ref_index = None
        for match in re.finditer(r'\[\d+\]', answer):
            last_ref_index = match.end()
    
        if not refs:
            return answer
    
        refs_str = ''.join(refs)

        def simplify(sentence, refs, docs):
            ref_numbers = [int(ref.strip('[]')) for ref in refs]
            
            docs_combined = ''.join(docs[ref - 1] for ref in ref_numbers if ref - 1 < len(docs))
            
            if not self.entail(docs_combined, sentence):
                return ''.join(refs)  
            
            if len(ref_numbers) == 1:
                return ''.join(f'[{num}]' for num in ref_numbers)
            def remove_and_test(ref_numbers):
                for i, ref in enumerate(ref_numbers):
                    new_ref_numbers = ref_numbers[:i] + ref_numbers[i+1:]
                    new_docs_combined = ''.join(docs[r - 1] for r in new_ref_numbers if r - 1 < len(docs))
                    if self.entail(new_docs_combined, sentence):
                        if len(new_ref_numbers) == 1:
                            return new_ref_numbers
                        return remove_and_test(new_ref_numbers)
                return ref_numbers

            simplified_ref_numbers = remove_and_test(ref_numbers)
            
            simplified_refs = ''.join(f'[{num}]' for num in simplified_ref_numbers)
            return simplified_refs

        simplified_refs = simplify(answer,refs,docs)

        sentence_without_refs = re.sub(r'\[\d+\]', '', answer)
        
        if last_ref_index is not None:
            output = sentence_without_refs[:last_ref_index - len(refs_str)] + simplified_refs + sentence_without_refs[last_ref_index - len(refs_str):]
        else:
            output =  sentence_without_refs + simplified_refs

        if self.multi_process:
            self.last_message.append(output)
        else:
            self.last_message = output
        
        return output


class Verifier(Module):

    '''
    Verifier is currently only used for single sentence/single target answer, not for parallel or iterative answer.
    Verifier will return dynamic prompt, not like other modules returning output. It is a judger only to decide the target module.
    By default, the verifoer verifies whether the documents support the answer.
    '''
    model_type = 'verifier'
    def __init__(self, prompt_maker = None, pipeline=None, self_prompt={}, criteria = None, key = 'answer', test = False) -> None:
        Module.__init__(self,prompt_maker, pipeline, self_prompt)
        if not test: 
            self.entail = _run_nli_autoais
        else:
            self.entail = lambda p,c : random.randint(0,1)
            self.docs = ['s'] * 100
        self.key = key
        self.test = test

    # Overcite this function to 
    def verifier_judge(self,**kargs):
        docs = self.pipeline.doc_cache
        answer = kargs[self.key]
        refs = re.findall(r'\[\d+\]', answer)
        if not refs:
            return False
        ref_numbers = [int(ref.strip('[]')) for ref in refs]
        
        docs_combined = ''.join(docs[ref - 1] for ref in ref_numbers if ref - 1 < len(docs))
        return bool(self.entail(docs_combined, re.sub(r'\[\d+\]', '', answer)))


    def generate(self, head_prompt: dict = {}, dynamic_prompt: dict = {}):
        prompt = {**head_prompt, **dynamic_prompt}
        out = self.verifier_judge(**prompt)

        self.last_message = out

        self.turns += 1
        return dynamic_prompt


class AugmentCluster():
    def __init__(self, module_list = []) -> None:
        self.module_list = module_list
        module_count = len(module_list)
        for i in range(module_count - 1):
            assert isinstance(module_list[i],LLM) and isinstance(module_list[i+1],LLM)
            module_list[i].set_target(module_list[i+1], post_processing = module_list[i].post_processing)
    
    def get_first_module(self):
        return self.module_list[0]
    
    def reset(self):
        for module in self.module_list:
            module.reset()

    def set_target(self,destination, condition = lambda self: True, post_processing = lambda x: x) -> None:
        self.module_list[-1].set_target(destination, condition, post_processing)
    
    def set_output(self, cond = lambda self: True, post_processing = lambda x:x, end = True):
        self.module_list[-1].set_output(cond, post_processing, end)

    def connect_to(self, pipeline = None) -> None:
        for module in self.module_list:
            module.connect_to(pipeline)


class Attribute_post_select(LLM):
    noisy = False
    model_name = 'function'
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
    
class Attribute_post_cluster(LLM):
    noisy = False
    model_name = 'function'
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
        return [{'span': _form(cls),'span_list': Prompt.UNABLE,'doc_map': Prompt.UNABLE,'clusters':Prompt.UNABLE,'docs':Prompt.UNABLE} for cls in clusters if _form(cls)]

class Ranker(EvalModule):
    
    def __init__(self, prompt_maker=None, pipeline=None, self_prompt={}, criteria=None,iterative = True, max_turn = 3, parallel = False) -> None:
        super().__init__(prompt_maker, pipeline, self_prompt, criteria, iterative = iterative, max_turn = max_turn, parallel = parallel)
        self.compare = []
    def generate(self, head_prompt: dict = {}, dynamic_prompt: dict = {}):
        
        self.turns += 1
        result = {}
        p_data = {**head_prompt, **self.self_prompt,**dynamic_prompt}
        for criteria, get_data in self.get_data.items():
            data_dict = {}
            for k, v in get_data.items():
                if v == 'doc_cache':
                    data_dict[k] = self.pipeline.doc_cache.show_docs()
                else:
                    if v in p_data.keys():
                        data_dict[k] = p_data[v]
                    else:
                        data_dict[k] = self.pipeline.current_data[v]
            eval_func = self.eval_criteria[criteria]
            data = [data_dict]
            result[criteria] = eval_func(data)
        
        result = sum([value for key, value in result.items()])
        self.compare.append((result,dynamic_prompt))
        output = max(self.compare,key = lambda x:x[0])[1]
        destination = self.send()
        self.last_message = output
        
        if destination in self.conditions:
            return self.conditions[destination]['post_processing'](output)
        else: 
            return self.post_processing(output)

        return {}

    def end_multi(self):
        self.compare = []
        return super().end_multi()
    


class AttributingModule(AugmentCluster):
    demo ={
   "selection_instruction":"In this task, you are presented with several documents, which need to be summarized. As an intermediate step, you need to identify salient content within the documents. For each document, copy verbatim the salient spans, and use <SPAN_DELIM> as a delimiter between each consecutive span. IMPORTANT: The output must be of the format Document [<DOC_ID>]: <SPAN_DELIM>-delimited consecutive salient spans. IMPORTANT: Each salient content must be a single consecutive verbatim span from the corresponding passages. IMPORTANT: make sure the total number of copied words (from all documents) is around 200 words, and not more than 900.",
    "selection_shot":"Document [1]: Cherrapunji Cherrapunji ( with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861, however: it received in\" \nDocument [2]: Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. Cherrapunji has often been credited as being the wettest place on Earth, but for now nearby Mawsynram currently holds that distinction. Cherrapunji still holds the all-time record for the most rainfall\" \nDocument [3]: \"Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in north-eastern India, 65 kilometres from Shillong. Mawsynram receives one of the highest rainfalls in India. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 mm, but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm between 1952 and 1989 and López de Micay, also in Colombia, which reported an annual 12,892 mm per year between 1960 and 2012. According to the \"Guinness Book of World Records\", Mawsynram received of rainfall in 1985. Mawsynram is located at 25° 18′\" \n\nAnswer: \nDocument [1]: <SPAN_DELIM>Cherrapunji has often been credited as being the wettest place on Earth<SPAN_DELIM> still holds the all-time record for the most rainfall in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861<SPAN_DELIM> \nDocument [2]: <SPAN_DELIM>Cherrapunji has often been credited as being the wettest place on Earth<SPAN_DELIM>still holds the all-time record for the most rainfall<SPAN_DELIM> \nDocument [3]: <SPAN_DELIM>Mawsynram receives one of the highest rainfalls in India <SPAN_DELIM> but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm between 1952 and 1989 <SPAN_DELIM> López de Micay, also in Colombia, which reported an annual 12,892 mm per year between 1960 and 2012. <SPAN_DELIM>",
    "clustering_instruction":"In this task, you are presented with several passages, where some parts are \"highlighted\" (namely, there are <highlight_start> and <highlight_end> tokens before and after each such span). The goal is to fuse all those highlights into a single summary. As an intermediate step, you need to cluster highlights that can be merged into a sentence (namely, each cluster will be later merged into one sentence). Make sure the clusters are in the same order you would then write the corresponding summary sentences. IMORTANT: make sure there are at most 3 clusters, and no more than 3 highlights per cluster. IMPORTANT: The output must be of the format [\"cluster\":[comma-delimited highlights indices]]",
    "clustering_shot":"Document [1]: Cherrapunji Cherrapunji ( with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. <highlight_start>Cherrapunji has often been credited as being the wettest place on Earth<highlight_end>, but for now nearby Mawsynram currently holds that distinction. Cherrapunji <highlight_start>still holds the all-time record for the most rainfall in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861<highlight_end>, however: it received in\" \nDocument [2]: Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. <highlight_start>Cherrapunji has often been credited as being the wettest place on Earth<highlight_end>, but for now nearby Mawsynram currently holds that distinction. <highlight_start>Cherrapunji still holds the all-time record for the most rainfall<highlight_end>\" \nDocument [3]: \"Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in north-eastern India, 65 kilometres from Shillong. <highlight_start>Mawsynram receives one of the highest rainfalls in India<highlight_end>. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 mm, <highlight_start>but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm between 1952 and 1989<highlight_end> and <highlight_start>López de Micay, also in Colombia, which reported an annual 12,892 mm per year between 1960 and 2012.<highlight_end> According to the \"Guinness Book of World Records\", Mawsynram received of rainfall in 1985. Mawsynram is located at 25° 18′\" \n\nThe highlighted spans are: \nDocument [1]: 1. Cherrapunji has often been credited as being the wettest place on Earth \n2. still holds the all-time record for the most rainfall in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861 \nDocument [2]: \n3. Cherrapunji has often been credited as being the wettest place on Earth \n4. still holds the all-time record for the most rainfall \nDocument [3]: \n5.  Mawsynram receives one of the highest rainfalls in India  \n6. but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm between 1952 and 1989  \n7. López de Micay, also in Colombia, which reported an annual 12,892 mm per year between 1960 and 2012. \n\nAnswer: \nThe highlighted spans are clustered as follows: \n[{\"cluster\":[6,7]}, {\"cluster\":[5]},{\"cluster\":[1,2]}]",
    "gen_instruction":"In this task, you are presented with several passages, where some parts are \"highlighted\" (namely, there are <highlight_start> and <highlight_end> tokens before and after each such span). You may also be presented with a prefix of the answer. You job is to generate the next sentence of the answer, that covers all and only the \"highlighted\" spans. Make sure it connects well with the prefix(if eixists), and that it covers all and only the \"highlighted\" spans. Always cite for any factual claim. When citing several search results, use [1][2][3]. Cite at least one document and at most three documents in each sentence. If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.",
    "gen_shot":"Document [1]: Cherrapunji Cherrapunji ( with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. <highlight_start>Cherrapunji has often been credited as being the wettest place on Earth<highlight_end>, but for now nearby Mawsynram currently holds that distinction. Cherrapunji <highlight_start>still holds the all-time record for the most rainfall in a calendar month for July 1861 and most rain in a year from August 1860 to July 1861<highlight_end>, however: it received in\" \nDocument [2]: Radio relay station known as Akashvani Cherrapunji. It broadcasts on FM frequencies. Cherrapunji Cherrapunji (; with the native name Sohra being more commonly used, and can also be spelled Cherrapunjee or Cherrapunji) is a subdivisional town in the East Khasi Hills district in the Indian state of Meghalaya. It is the traditional capital of aNongkhlaw \"hima\" (Khasi tribal chieftainship constituting a petty state), both known as Sohra or Churra. <highlight_start>Cherrapunji has often been credited as being the wettest place on Earth<highlight_end>, but for now nearby Mawsynram currently holds that distinction. <highlight_start>Cherrapunji still holds the all-time record for the most rainfall<highlight_end>\" \nDocument [3]: \"Mawsynram Mawsynram () is a village in the East Khasi Hills district of Meghalaya state in north-eastern India, 65 kilometres from Shillong. <highlight_start>Mawsynram receives one of the highest rainfalls in India<highlight_end>. It is reportedly the wettest place on Earth, with an average annual rainfall of 11,872 mm, <highlight_start>but that claim is disputed by Lloró, Colombia, which reported an average yearly rainfall of 12,717 mm between 1952 and 1989<highlight_end> and <highlight_start>López de Micay, also in Colombia, which reported an annual 12,892 mm per year between 1960 and 2012.<highlight_end> According to the \"Guinness Book of World Records\", Mawsynram received of rainfall in 1985. Mawsynram is located at 25° 18′\" \n\nPrefix: Several places on Earth claim to be the most rainy, such as Lloró, Colombia, which reported an average annual rainfall of 12,717 mm between 1952 and 1989, and López de Micay, Colombia, which reported an annual 12,892 mm between 1960 and 2012 [3].  \n\nThe highlighted spans are: \nDocument [3]: \n5.  Mawsynram receives one of the highest rainfalls in India \n\nAnswer: \nThe next sentence is: \nHowever, the official record is held by Mawsynram, India with an average annual rainfall of 11,872 mm [3]."
    }
    PARA_SEP = '\n\n'
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
    def __init__(self, model) -> None:
        module_list = []
        select = LLM(model = model, prompt_maker = self.prompt, self_prompt={'INST':self.demo['selection_instruction'],'shot':self.selection_shot,'add':''}, post_processing=make_as('span'),noisy= False)
        post_select = Attribute_post_select()
        clustering =  LLM(model = model, prompt_maker = self.prompt, self_prompt={'INST':self.demo['clustering_instruction'],'shot':self.cls_shot, 'add':'The highlighted spans are clustered as follows:'},share_model_with=select, post_processing=make_as('cls'),noisy=False)
        post_cls = Attribute_post_cluster()
        module_list = [select,post_select,clustering,post_cls]
        super().__init__(module_list)
