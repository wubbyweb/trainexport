"""
train_and_export.py  (numpy version — same algorithm as microgpt, vectorized)
Run:  python train_and_export.py
Output: intent_weights.json
"""
import math, random, json, os
import numpy as np

random.seed(42); np.random.seed(42)

# ── DATASET ──────────────────────────────────────────────────────────────────
raw_docs = [
    "S|what did we discuss about the new project",
    "S|summarize our conversation about the API design",
    "S|what was decided about the database schema",
    "S|find messages about the budget approval",
    "S|what did Alice say about the deployment plan",
    "S|any discussion about security vulnerabilities",
    "S|what did the team say about the roadmap",
    "S|find anything related to the client feedback",
    "S|what were the concerns raised about performance",
    "S|did anyone mention the pipeline issues",
    "S|what was the outcome of the architecture review",
    "S|summarize what was said about onboarding",
    "S|find messages about the quarterly goals",
    "S|what did Bob say about the new hire",
    "S|any updates on the authentication bug",
    "S|what was discussed about the migration strategy",
    "S|find our conversation about caching",
    "S|what did we agree on for the sprint planning",
    "S|any messages about the frontend redesign",
    "S|what was said about the vendor contract",
    "S|find anything about the data privacy policy",
    "S|what did the team think about using GraphQL",
    "S|any discussion about scaling the backend",
    "S|what was mentioned about test coverage",
    "S|find messages about the product launch timeline",
    "S|what did we say about containerization",
    "S|any conversation about the API rate limits",
    "S|what was the feedback on the prototype demo",
    "S|find messages about team velocity",
    "S|what did we discuss about logging and monitoring",
    "S|what was said about the microservices architecture",
    "S|find anything about the user onboarding flow",
    "S|what did we decide about database indexing",
    "S|any messages about the code review process",
    "S|what was discussed regarding error handling",
    "D|messages from January 31 2026",
    "D|what was said on March 5th",
    "D|show me messages from last Tuesday",
    "D|find everything from February 14",
    "D|what happened on December 25th",
    "D|messages sent on 2026-01-15",
    "D|what did we talk about on Monday",
    "D|show conversations from yesterday",
    "D|find messages from the 3rd of this month",
    "D|what was discussed on April 1st",
    "D|messages from last Friday",
    "D|what happened on the 10th of January",
    "D|show me what was said on Thursday",
    "D|find messages from two days ago",
    "D|what was talked about on November 22",
    "D|messages from 01/20/2026",
    "D|what did we discuss on the 15th",
    "D|show everything from last Wednesday",
    "D|find messages from the beginning of the month",
    "D|what was said on New Years Day",
    "D|messages from March 10th",
    "D|what happened last Sunday",
    "D|show me messages from yesterday afternoon",
    "D|find the conversation from last Monday",
    "D|what was discussed on February 28",
    "D|messages sent on the 7th",
    "D|what did people say on Christmas Eve",
    "D|show conversations from 3 days ago",
    "D|find messages from the start of last week",
    "D|what was said on January 1st",
    "D|messages from last Saturday",
    "D|what happened on the morning of the 4th",
    "D|find conversations from this past Monday",
    "D|messages from the 22nd of last month",
    "D|what was discussed on the afternoon of March 3rd",
    "R|show me the last 5 messages",
    "R|what are the most recent messages",
    "R|list the latest 10 conversations",
    "R|show the newest messages",
    "R|what was the last thing said",
    "R|give me the most recent 3 messages",
    "R|show the last message in the chat",
    "R|what are the latest updates",
    "R|list the 5 most recent conversations",
    "R|show the last 20 messages",
    "R|what was most recently discussed",
    "R|give me the newest 7 messages",
    "R|show me recent activity",
    "R|what happened most recently",
    "R|list the latest messages",
    "R|show the last few things said",
    "R|what are the most recent 15 messages",
    "R|give me the last message",
    "R|show me the newest conversation",
    "R|what was the most recent update",
    "R|list the last 10 items",
    "R|show recent messages from the team",
    "R|what are the latest 5 entries",
    "R|give me the most recent activity",
    "R|show the last 3 conversations",
    "R|what happened in the last message",
    "R|list the newest messages first",
    "R|show me the last 8 messages",
    "R|what are the most recent discussions",
    "R|give me the latest 12 messages",
    "R|show the most recent 25 messages",
    "R|what was the last update from the team",
    "R|give me the newest 15 entries",
    "R|show the last 50 messages",
    "R|what are the latest 20 conversations",
]
docs = raw_docs[:]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# ── TOKENIZER ─────────────────────────────────────────────────────────────────
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
c2i = {c: i for i, c in enumerate(uchars)}
print(f"vocab size: {vocab_size}")

def tokenize(doc):
    return [BOS] + [c2i[c] for c in doc] + [BOS]

# ── HYPERPARAMS ───────────────────────────────────────────────────────────────
n_embd=16; n_head=4; n_layer=2; block_size=64; head_dim=n_embd//n_head

# ── WEIGHTS ───────────────────────────────────────────────────────────────────
def mat(r,c,s=0.08): return np.random.randn(r,c).astype(np.float64)*s
W = {'wte':mat(vocab_size,n_embd),'wpe':mat(block_size,n_embd),'lm_head':mat(vocab_size,n_embd)}
for i in range(n_layer):
    W[f'l{i}.wq']=mat(n_embd,n_embd); W[f'l{i}.wk']=mat(n_embd,n_embd)
    W[f'l{i}.wv']=mat(n_embd,n_embd); W[f'l{i}.wo']=mat(n_embd,n_embd)
    W[f'l{i}.fc1']=mat(4*n_embd,n_embd); W[f'l{i}.fc2']=mat(n_embd,4*n_embd)
print(f"num params: {sum(v.size for v in W.values())}")

# ── FORWARD ───────────────────────────────────────────────────────────────────
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

# ── ADAM ──────────────────────────────────────────────────────────────────────
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

# ── EXPORT ────────────────────────────────────────────────────────────────────
export={
    "config":{"n_embd":n_embd,"n_head":n_head,"n_layer":n_layer,
              "block_size":block_size,"vocab_size":vocab_size,"head_dim":head_dim},
    "tokenizer":{"uchars":uchars,"BOS":BOS},
    "labels":{"S":"semantic","D":"date-specific","R":"recency"},
    "weights":{k:v.tolist() for k,v in W.items()},
}
with open("intent_weights.json","w") as f: json.dump(export,f,separators=(',',':'))
kb=os.path.getsize("intent_weights.json")/1024
print(f"Weights saved → intent_weights.json ({kb:.1f} KB)")

# ── SANITY CHECK ──────────────────────────────────────────────────────────────
def score(doc_str):
    toks=tokenize(doc_str); src=toks[:-1][:block_size]; tgt=np.array(toks[1:len(src)+1])
    logits,_=forward(src); p=softmax_np(logits)
    return -np.log(p[np.arange(len(tgt)),tgt]+1e-9).mean()

def classify(q):
    sc={k:score(f"{k}|{q}") for k in ['S','D','R']}
    best=min(sc,key=sc.get)
    return {'S':'semantic','D':'date-specific','R':'recency'}[best],sc

tests=[
    ("what did we decide about the infrastructure upgrade","semantic"),
    ("messages from February 3rd","date-specific"),
    ("show me the last 5 messages","recency"),
    ("find anything about the login bug","semantic"),
    ("what was said on Monday","date-specific"),
    ("give me the most recent messages","recency"),
    ("any discussion about the pricing model","semantic"),
    ("messages from last Thursday","date-specific"),
    ("list the latest 10 conversations","recency"),
]
print("\nSanity check:")
ok=0
for q,exp in tests:
    pred,sc=classify(q); tick="✓" if pred==exp else "✗"
    print(f"  {tick} {pred:13s} ← \"{q}\""); ok+=pred==exp
print(f"  accuracy: {ok}/{len(tests)} = {100*ok//len(tests)}%")
print("\nNext: open intent_classifier.html in a browser alongside intent_weights.json")