import argparse
import csv
import json
import os
import time
import pickle

import numpy as np
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def gtr_build_index(encoder, docs):
    with torch.inference_mode():
        embs = encoder.encode(docs, show_progress_bar=True, normalize_embeddings=True)
        embs = embs.astype("float16")

    GTR_EMB = os.environ.get("GTR_EMB")
    with open(GTR_EMB, "wb") as f:
        pickle.dump(embs, f)
    return embs


class DPRRetriever:
    def __init__(self, DPR_WIKI_TSV, GTR_EMB = None, emb_model = "sentence-transformers/gtr-t5-xxl") -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.encoder = SentenceTransformer(emb_model, device = device)
        self.docs = []
        print("loading wikipedia file...")
        with open(DPR_WIKI_TSV) as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.docs.append(row[2] + "\n" + row[1])
        if not GTR_EMB:
            print("gtr embeddings not found, building...")
            embs = gtr_build_index(self.encoder, self.docs)
        else:
            print("gtr embeddings found, loading...")
            with open(GTR_EMB, "rb") as f:
                embs = pickle.load(f)

        self.gtr_emb = torch.tensor(embs, dtype=torch.float16, device=device)

    def retrieve(self, question, topk):
        with torch.inference_mode():
            query = self.encoder.encode(question, batch_size=4, show_progress_bar=True, normalize_embeddings=True)
            query = torch.tensor(query, dtype=torch.float16, device=self.device)
        query = query.to(self.device)
        scores = torch.matmul(self.gtr_emb, query)
        score, idx = torch.topk(scores, topk)
        ret = []
        for i in range(idx.size(0)):
            title, text = self.docs[idx[i].item()].split("\n")
            ret.append({"id": str(idx[i].item() + 1), "title": title, "text": text, "score": score[i].item()})
        
        return ret
    
    def __repr__(self) -> str:
        return 'DPR Retriever'
    
    def __str__(self) -> str:
        return repr(self)

class BM25Retriever:
    def __init__(self, DPR_WIKI_TSV):
        self.docs = []
        self.tokenized_docs = []
        print("loading wikipedia file...")
        with open(DPR_WIKI_TSV) as f:
            reader = csv.reader(f, delimiter="\t")
            for i, row in enumerate(reader):
                if i == 0:
                    continue
                self.docs.append(row[2] + "\n" + row[1])
                self.tokenized_docs.append((row[2] + " " + row[1]).split())

        print("BM25 index building...")
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, question, topk):
        query = question.split()
        scores = self.bm25.get_scores(query)
        topk_indices = scores.argsort()[-topk:][::-1]
        ret = []
        for idx in topk_indices:
            title, text = self.docs[idx].split("\n", 1)
            ret.append({"id": str(idx + 1), "title": title, "text": text, "score": scores[idx]})
        
        return ret
    def __repr__(self) -> str:
        return 'BM25 Retriever'
    
    def __str__(self) -> str:
        return repr(self)
    