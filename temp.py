"""
train_and_export.py  (numpy version — same algorithm as microgpt, vectorized)
Run:  python train_and_export.py
Output: intent_weights.json
"""
import math, random, json, os
import numpy as np

random.seed(42); np.random.seed(42)
raw_docs = []
categories = set()
with open('source/financedata.csv', 'r') as f:
    for line in f:
        line = line.strip().strip('"')
        if not line: continue
        parts = line.rsplit(',', 1)
        if len(parts) == 2:
            desc, cat = parts
            raw_docs.append(f"{cat}|{desc}")
            categories.add(cat)
categories = sorted(list(categories))
docs = raw_docs[:]
import random
random.shuffle(docs)
print(f"num docs: {len(docs)}")
print(f"num categories: {len(categories)}")
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
c2i = {c: i for i, c in enumerate(uchars)}
print(f"vocab size: {vocab_size}")

def tokenize(doc):
    print([BOS])
    return [BOS] + [c2i[c] for c in doc] + [BOS]
n_embd=16; n_head=4; n_layer=2; block_size=64; head_dim=n_embd//n_head
def mat(r,c,s=0.08): return np.random.randn(r,c).astype(np.float64)*s
W = {'wte':mat(vocab_size,n_embd),'wpe':mat(block_size,n_embd),'lm_head':mat(vocab_size,n_embd)}
for i in range(n_layer):
    W[f'l{i}.wq']=mat(n_embd,n_embd); W[f'l{i}.wk']=mat(n_embd,n_embd)
    W[f'l{i}.wv']=mat(n_embd,n_embd); W[f'l{i}.wo']=mat(n_embd,n_embd)
    W[f'l{i}.fc1']=mat(4*n_embd,n_embd); W[f'l{i}.fc2']=mat(n_embd,4*n_embd)
print(f"num params: {sum(v.size for v in W.values())}")
def softmax_np(x,axis=-1):
    x=x-x.max(axis=axis,keepdims=True); e=np.exp(x); return e/e.sum(axis=axis,keepdims=True)
def rmsnorm(x): return x/np.sqrt(np.mean(x**2,axis=-1,keepdims=True)+1e-5)

def forward(tokens):
    T=len(tokens); tids=np.array(tokens); pids=np.arange(T)
    x=rmsnorm(W['wte'][tids]+W['wpe'][pids])
    cache={'x_layers':[x.copy()]}
    for li in range(n_layer):
        xr=x; xn=rmsnorm(x)
        Q=xn@W[f'l{li}.wq'].T; K=xn@W[f'l{li}.wk'].T; V=xn@W[f'l{li}.wv'].T
        Qh=Q.reshape(T,n_head,head_dim).transpose(1,0,2)
        Kh=K.reshape(T,n_head,head_dim).transpose(1,0,2)
        Vh=V.reshape(T,n_head,head_dim).transpose(1,0,2)
        sc=Qh@Kh.transpose(0,2,1)/math.sqrt(head_dim)
        sc+=np.triu(np.full((T,T),-1e9),k=1)
        aw=softmax_np(sc,axis=-1)
        ho=(aw@Vh).transpose(1,0,2).reshape(T,n_embd)
        x=xr+ho@W[f'l{li}.wo'].T
        xr2=x; xn2=rmsnorm(x)
        h=np.maximum(0,xn2@W[f'l{li}.fc1'].T)
        x=xr2+h@W[f'l{li}.fc2'].T
        cache['x_layers'].append(x.copy())
        cache[f'xn{li}']=xn; cache[f'aw{li}']=aw
        cache[f'Q{li}']=Q; cache[f'K{li}']=K; cache[f'V{li}']=V
        cache[f'xn2_{li}']=xn2; cache[f'h{li}']=h; cache[f'xr{li}']=xr; cache[f'xr2_{li}']=xr2
    logits=x@W['lm_head'].T; cache['xf']=x
    return logits, cache

def loss_and_grad(logits, tgt):
    p=softmax_np(logits); ll=-np.log(p[np.arange(len(tgt)),tgt]+1e-9).mean()
    dl=p.copy(); dl[np.arange(len(tgt)),tgt]-=1; dl/=len(tgt)
    return ll, dl

def backward(tokens, cache, dlogits):
    T=len(tokens); tids=np.array(tokens); pids=np.arange(T)
    G={k:np.zeros_like(v) for k,v in W.items()}
    xf=cache['xf']; G['lm_head']+=dlogits.T@xf; dx=dlogits@W['lm_head']
    for li in reversed(range(n_layer)):
        xn2=cache[f'xn2_{li}']; h=cache[f'h{li}']; xr2=cache[f'xr2_{li}']
        G[f'l{li}.fc2']+=dx.T@h; dh=dx@W[f'l{li}.fc2']; dh*=(h>0)
        G[f'l{li}.fc1']+=dh.T@xn2
        rms2=np.sqrt(np.mean(xr2**2,axis=-1,keepdims=True)+1e-5)
        dx=dx+dh@W[f'l{li}.fc1']/rms2
        xn=cache[f'xn{li}']; aw=cache[f'aw{li}']
        Q=cache[f'Q{li}']; K=cache[f'K{li}']; V=cache[f'V{li}']; xr=cache[f'xr{li}']
        Vh=V.reshape(T,n_head,head_dim).transpose(1,0,2)
        Qh=Q.reshape(T,n_head,head_dim).transpose(1,0,2)
        Kh=K.reshape(T,n_head,head_dim).transpose(1,0,2)
        ho=(aw@Vh).transpose(1,0,2).reshape(T,n_embd)
        G[f'l{li}.wo']+=dx.T@ho; dx_attn=dx@W[f'l{li}.wo']
        dxh=dx_attn.reshape(T,n_head,head_dim).transpose(1,0,2)
        dVh=aw.transpose(0,2,1)@dxh; daw=dxh@Vh.transpose(0,2,1)
        dsc=aw*(daw-(daw*aw).sum(-1,keepdims=True)); dsc/=math.sqrt(head_dim)
        dQh=dsc@Kh; dKh=dsc.transpose(0,2,1)@Qh
        dQ=dQh.transpose(1,0,2).reshape(T,n_embd)
        dK=dKh.transpose(1,0,2).reshape(T,n_embd)
        dV=dVh.transpose(1,0,2).reshape(T,n_embd)
        G[f'l{li}.wq']+=dQ.T@xn; G[f'l{li}.wk']+=dK.T@xn; G[f'l{li}.wv']+=dV.T@xn
        dxn=(dQ@W[f'l{li}.wq']+dK@W[f'l{li}.wk']+dV@W[f'l{li}.wv'])
        rms_a=np.sqrt(np.mean(xr**2,axis=-1,keepdims=True)+1e-5)
        dx=dx+dxn/rms_a
    emb=W['wte'][tids]+W['wpe'][pids]
    rms0=np.sqrt(np.mean(emb**2,axis=-1,keepdims=True)+1e-5)
    de=dx/rms0
    np.add.at(G['wte'],tids,de); np.add.at(G['wpe'],pids,de)
    return G
lr=0.003; b1=0.9; b2=0.999; eps=1e-8
mA={k:np.zeros_like(v) for k,v in W.items()}
vA={k:np.zeros_like(v) for k,v in W.items()}
num_steps=5000
print(f"\nTraining {num_steps} steps...")
for step in range(num_steps):
    doc=docs[step%len(docs)]; toks=tokenize(doc)
    src=toks[:-1][:block_size]; tgt=np.array(toks[1:len(src)+1])
    logits,cache=forward(src); loss,dl=loss_and_grad(logits,tgt)
    G=backward(src,cache,dl)
    lr_t=lr*(1-step/num_steps); t=step+1
    for k in W:
        mA[k]=b1*mA[k]+(1-b1)*G[k]; vA[k]=b2*vA[k]+(1-b2)*G[k]**2
        mh=mA[k]/(1-b1**t); vh=vA[k]/(1-b2**t)
        W[k]-=lr_t*mh/(np.sqrt(vh)+eps)
    if (step+1)%1000==0: print(f"  step {step+1:5d}/{num_steps} | loss {loss:.4f}")
print(f"Done. Final loss: {loss:.4f}")
export = {
    "config": {
        "n_embd": n_embd, "n_head": n_head, "n_layer": n_layer,
        "block_size": block_size, "vocab_size": vocab_size, "head_dim": head_dim
    },
    "tokenizer": {"uchars": uchars, "BOS": BOS},
    "labels": categories,
    "weights": {k: v.tolist() for k, v in W.items()},
}

# Explicitly create the full path in the current working directory
save_path = os.path.join(os.getcwd(), "intent_weights.json")

with open(save_path, "w") as f:
    json.dump(export, f, separators=(',', ':'))

# Use the same save_path variable to check the file size and print the message
kb = os.path.getsize(save_path) / 1024
print(f"Weights saved → {save_path} ({kb:.1f} KB)")

def score(doc_str):
    toks=tokenize(doc_str); src=toks[:-1][:block_size]; tgt=np.array(toks[1:len(src)+1])
    logits,_=forward(src); p=softmax_np(logits)
    return -np.log(p[np.arange(len(tgt)),tgt]+1e-9).mean()

def classify(q):
    sc={k:score(f"{k}|{q}") for k in categories}
    best=min(sc,key=sc.get)
    return best, sc

tests=[
    ("PILOT_00036 00036 VALPARAISO IN", "Gas"),
    ("UDIPI PALACE SCHAUMBURG IL", "Food & Dining"),
    ("UBER   *TRIP", "Rental Car & Taxi")
]
print("\nSanity check:")
ok=0
for q,exp in tests:
    pred,sc=classify(q); tick="✓" if pred==exp else "✗"
    print(f"  {tick} {pred[:13]:13s} ← \"{q}\""); ok+=pred==exp
print(f"  accuracy: {ok}/{len(tests)} = {100*ok//len(tests)}%")

import json
import numpy as np

# 1. Load the exported JSON data
with open("intent_weights.json", "r") as f:
    loaded = json.load(f)

# 2. Reconstruct the weights and metadata
W_l = {k: np.array(v) for k, v in loaded['weights'].items()}
uchars_l = loaded['tokenizer']['uchars']
c2i_l = {c: i for i, c in enumerate(uchars_l)}
BOS_l = loaded['tokenizer']['BOS']
config_l = loaded['config']

# 3. Helper functions using the loaded weights
def tokenize_l(doc):
    return [BOS_l] + [c2i_l[c] for c in doc] + [BOS_l]

def forward_l(tokens):
    T=len(tokens); tids=np.array(tokens); pids=np.arange(T)
    x=rmsnorm(W_l['wte'][tids] + W_l['wpe'][pids])
    for li in range(config_l['n_layer']):
        xn=rmsnorm(x); xr=x
        Q=xn@W_l[f'l{li}.wq'].T; K=xn@W_l[f'l{li}.wk'].T; V=xn@W_l[f'l{li}.wv'].T
        Qh=Q.reshape(T, config_l['n_head'], config_l['head_dim']).transpose(1,0,2)
        Kh=K.reshape(T, config_l['n_head'], config_l['head_dim']).transpose(1,0,2)
        Vh=V.reshape(T, config_l['n_head'], config_l['head_dim']).transpose(1,0,2)
        sc=Qh@Kh.transpose(0,2,1)/math.sqrt(config_l['head_dim'])
        sc+=np.triu(np.full((T,T),-1e9),k=1)
        aw=softmax_np(sc,axis=-1)
        ho=(aw@Vh).transpose(1,0,2).reshape(T, config_l['n_embd'])
        x=xr+ho@W_l[f'l{li}.wo'].T
        xn2=rmsnorm(x); xr2=x
        h=np.maximum(0, xn2@W_l[f'l{li}.fc1'].T)
        x=xr2+h@W_l[f'l{li}.fc2'].T
    return x@W_l['lm_head'].T

def classify_l(q):
    def score_l(doc_str):
        toks=tokenize_l(doc_str); src=toks[:-1][:config_l['block_size']]; tgt=np.array(toks[1:len(src)+1])
        logits=forward_l(src); p=softmax_np(logits)
        return -np.log(p[np.arange(len(tgt)),tgt]+1e-9).mean()
    
    sc={k:score_l(f"{k}|{q}") for k in categories}
    best=min(sc,key=sc.get)
    return best, sc

# 4. Run sample classifications using ONLY the loaded JSON weights
sample_queries = [
    "PILOT_00036 00036 VALPARAISO IN",
    "UDIPI PALACE SCHAUMBURG IL",
]
categories = loaded.get('labels', [])

print("Testing classification using intent_weights.json:\n")
for q in sample_queries:
    intent, scores = classify_l(q)
    print(f"Query: \"{q}\"")
    print(f"Result: {intent}\n")

