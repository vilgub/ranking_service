import string
import json
import os

from flask import Flask, request

from collections import Counter
from typing import Dict, List, Tuple, Union, Callable
from datetime import datetime
from langdetect import detect

import nltk
import numpy as np
import math
import pandas as pd
import torch
import torch.nn.functional as F
import faiss


class GaussianKernel(torch.nn.Module):
    def __init__(self, mu: float = 1., sigma: float = 1.):
        super().__init__()
        self.mu = mu
        self.sigma = sigma

    def forward(self, x):
        return torch.exp(
            -0.5 * ((x - self.mu) ** 2) / (self.sigma ** 2)
        )


class KNRM(torch.nn.Module):
    def __init__(self,
                 embedding_matrix: np.ndarray = None,
                 emb_path: str = '', mlp_path: str = '',
                 freeze_embeddings: bool = True, kernel_num: int = 21,
                 sigma: float = 0.1, exact_sigma: float = 0.001,
                 out_layers: List[int] = [10, 5]):
        super().__init__()


        tl = torch.load(emb_path)
        self.embeddings = torch.nn.Embedding(num_embeddings=tl['weight'].shape[0], embedding_dim=tl['weight'].shape[1])
        self.embeddings.load_state_dict(tl)

        self.kernel_num = kernel_num
        self.sigma = sigma
        self.exact_sigma = exact_sigma
        self.out_layers = out_layers

        self.kernels = self._get_kernels_layers()

        self.mlp = self._get_mlp()
        self.mlp.load_state_dict(torch.load(mlp_path))

        self.out_activation = torch.nn.Sigmoid()

    def _get_kernels_layers(self) -> torch.nn.ModuleList:
        kernels = torch.nn.ModuleList()
        for i in range(self.kernel_num):
            mu = 1. / (self.kernel_num - 1) + (2. * i) / (
                self.kernel_num - 1) - 1.0
            sigma = self.sigma
            if mu > 1.0:
                sigma = self.exact_sigma
                mu = 1.0
            kernels.append(GaussianKernel(mu=mu, sigma=sigma))
        return kernels

    def _get_mlp(self) -> torch.nn.Sequential:
        out_cont = [self.kernel_num] + self.out_layers + [1]
        mlp = [
            torch.nn.Sequential(
                torch.nn.Linear(in_f, out_f),
                torch.nn.ReLU()
            )
            for in_f, out_f in zip(out_cont, out_cont[1:])
        ]
        mlp[-1] = mlp[-1][:-1]
        return torch.nn.Sequential(*mlp)

    def forward(self, input_1: Dict[str, torch.Tensor], input_2: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        logits_1 = self.predict(input_1)
        logits_2 = self.predict(input_2)

        logits_diff = logits_1 - logits_2

        out = self.out_activation(logits_diff)
        return out

    def _get_matching_matrix(self, query: torch.Tensor, doc: torch.Tensor) -> torch.FloatTensor:

        return matching_matrix

    def _apply_kernels(self, matching_matrix: torch.FloatTensor) -> torch.FloatTensor:
        KM = []
        for kernel in self.kernels:
            # shape = [B]
            K = torch.log1p(kernel(matching_matrix).sum(dim=-1)).sum(dim=-1)
            KM.append(K)

        # shape = [B, K]
        kernels_out = torch.stack(KM, dim=1)
        return kernels_out

    def predict(self, inputs: Dict[str, torch.Tensor]) -> torch.FloatTensor:
        query, doc = inputs['query'], inputs['document']
        # shape = [B, L, R]
        matching_matrix = self._get_matching_matrix(query, doc)
        # shape [B, K]
        kernels_out = self._apply_kernels(matching_matrix)
        # shape [B]
        out = self.mlp(kernels_out)
        return out


class RankingDataset(torch.utils.data.Dataset):
    def __init__(self, index_pairs_or_triplets: List[List[Union[str, float]]],
                 idx_to_text_mapping: Dict[str, str], vocab: Dict[str, int], oov_val: int,
                 preproc_func: Callable, max_len: int = 30):
        self.index_pairs_or_triplets = index_pairs_or_triplets
        self.idx_to_text_mapping = idx_to_text_mapping
        self.vocab = vocab
        self.oov_val = oov_val
        self.preproc_func = preproc_func
        self.max_len = max_len

    def __len__(self):
        return len(self.index_pairs_or_triplets)

    def _tokenized_text_to_index(self, tokenized_text: List[str]) -> List[int]:
        res = [self.vocab.get(i, self.oov_val) for i in tokenized_text]
        return res

    def _convert_text_idx_to_token_idxs(self, idx: int) -> List[int]:
        tokenized_text = self.preproc_func(self.idx_to_text_mapping[idx])
        idxs = self._tokenized_text_to_index(tokenized_text)
        return idxs

    def __getitem__(self, idx: int):
        pass


EMB_PATH_KNRM  = os.environ['EMB_PATH_KNRM']
VOCAB_PATH     = os.environ['VOCAB_PATH']
EMB_PATH_GLOVE = os.environ['EMB_PATH_GLOVE']
MLP_PATH       = os.environ['MLP_PATH'] 


class OnlineServiceSolution():
    def __init__(self):
        self.is_prepare_service = False
        self.faiss_index_size = 0
    
    def _read_glove_embeddings(self, file_path: str) -> Dict[str, List[str]]:
        embedding_data = {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                current_line = line.rstrip().split(' ')
                embedding_data[current_line[0]] = current_line[1:]
                
        return embedding_data
    
    def prepare_service(self):
        self.KNRM = KNRM(
            emb_path=EMB_PATH_KNRM,
            mlp_path=MLP_PATH,
            freeze_embeddings=True,
            out_layers=[],
            kernel_num=21
        )
        
        with open(VOCAB_PATH, 'r') as read_file:
            self.vocab = json.load(read_file)
        
        self.glove_emb = self._read_glove_embeddings(EMB_PATH_GLOVE)
        
        self.is_prepare_service = True
        
        return True
    
    def prepare_faiss_indexes(self, documents: Dict[str, str]):
        oov_val = self.vocab["OOV"]
        
        self.documents = documents
        
        idxs, docs = [], []
        
        for idx in documents:
            idxs.append(int(idx))
            docs.append(documents[idx])
        
        embeddings = []
        emb_layer = self.model.embeddings.state_dict()['weight']
        
        for d in docs:
            tmp_emb = [self.vocab.get(w, oov_val) for w in self._simple_preproc(d)]
            tmp_emb = emb_layer[tmp_emb].mean(dim = 0)
            embeddings.append(np.array(tmp_emb))          
        
        embeddings = np.array([embedding for embedding in embeddings]).astype(np.float32)
        
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index = faiss.IndexIDMap(self.index)
        self.index.add_with_ids(embeddings, np.array(idxs))
        
        self.faiss_index_size = self.index.ntotal
        
        return self.faiss_index_size
    
    def _punctuation(self, inp_str: str) -> str:
        inp_str = str(inp_str)
        
        for punct in string.punctuation:
            inp_str = inp_str.replace(punct, ' ')
        
        return inp_str
    
    def _simple_preproc(self, inp_str: str):
        base_str = inp_str.strip().lower()
        str_wo_punct = self._punctuation(base_str)
        
        return nltk.word_tokenize(str_wo_punct)
    
    def _text_to_token_ids(self, text_list: List[str]):
        tokenized = []
        
        for text in text_list:
            tokenized_text = self._simple_preproc(text)
            token_idxs = [self.vocab.get(i, self.vocab["OOV"]) for i in tokenized_text]
            tokenized.append(token_idxs)
        
        max_len = max(len(elem) for elem in tokenized)
        
        tokenized = [elem + [0] * (max_len - len(elem)) for elem in tokenized]
        tokenized = torch.LongTensor(tokenized)    
        
        return tokenized
    
    def get_suggestion(self, 
            query: str, ret_k: int = 10, 
            ann_k: int = 100) -> List[Tuple[str, str]]:
        q_tokens = self._simple_preproc(query)
        vector = [self.vocab.get(tok, self.vocab["OOV"]) for tok in q_tokens]
        emb_layer = self.model.embeddings.state_dict()['weight']
        
        q_emb = emb_layer[vector].mean(dim = 0).reshape(1, -1)
        q_emb = np.array(q_emb).astype(np.float32)
        
        _, I = self.index.search(q_emb, k = ann_k)
        
        cands = [(str(i), self.documents[str(i)]) for i in I[0] if i != -1]
        
        inputs = dict()
        
        inputs['query'] = self._text_to_token_ids([query] * len(cands))
        inputs['document'] = self._text_to_token_ids([cnd[1] for cnd in cands])
        
        scores = self.model(inputs)
        
        res_ids = scores.reshape(-1).argsort(descending=True)
        res_ids = res_ids[:ret_k]
        
        res = [cands[i] for i in res_ids.tolist()]
        
        return res
    
    def query_proc(self, queries):
        lang_check = []
        suggestions = []
        
        for q in queries:
            is_en = (detect(q) == "en")
            lang_check.append(is_en)
            
            if not is_en:
                suggestions.append(None)
                continue
            
            suggestion = self.get_suggestion(q)
            suggestions.append(suggestion)
        
        return suggestions, lang_check


oss = OnlineServiceSolution()
app = Flask(__name__)


@app.route('/ping')
def ping():
    if not oss.is_prepare_service:
        return json.dumps({'status': 'not ok'})
    
    return json.dumps({'status': 'ok'})

@app.route('/query', methods=['POST'])
def query():
    if oss.faiss_index_size == 0:
        return json.dumps({'status': 'FAISS is not initialized!'})

    content = json.loads(request.json)
    queries = content['queries']

    suggestions, lang_check = oss.query_proc(queries)
    return json.dumps({'suggestions': suggestions, 'lang_check': lang_check})

@app.route('/update_index', methods=['POST'])
def update_index():
    content = json.loads(request.json)
    documents = content['documents']

    index_size = oss.prepare_faiss_indexes(documents)
    return json.dumps({'status': 'ok', 'index_size': index_size})


oss.prepare_service()
