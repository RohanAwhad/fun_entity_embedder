import os
import torch
torch.set_grad_enabled(False)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple

embedding_chkpt = os.getenv('EMBEDDING_CHKPT', None)
if embedding_chkpt is None: raise ValueError('EMBEDDING_CHKPT env var is not set')
tokenizer = AutoTokenizer.from_pretrained(embedding_chkpt)
model = AutoModel.from_pretrained(embedding_chkpt)
model.eval()

app = FastAPI()
app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials=True,
  allow_methods=['*'],
  allow_headers=['*']
)

class ReqIn(BaseModel):
  text: str
  query_entities_span: List[Tuple[int, int]]

class ResOut(BaseModel):
  entity_embeddings: List[List[float]]


def get_indices_within_offset(offset_mapping, start_offset, end_offset):
  return [
    i+1 for i, (start, end) in enumerate(offset_mapping[1:-1])
    if not (end < start_offset or start > end_offset)
  ]  # 1:-1 because of [CLS] token & [SEP] token

@app.post('/embed_entity')
async def embed_entity(req: ReqIn):
  inp = tokenizer(req.text, return_tensors='pt', return_offsets_mapping=True)
  offset_mapping = inp.pop('offset_mapping').tolist()[0]
  embeddings = model(**inp).last_hidden_state.squeeze(0)
  entity_token_indices = []
  for sf, ef in req.query_entities_span:
    entity_token_indices.append(get_indices_within_offset(offset_mapping, sf, ef))

  entity_embeddings = []
  for token_indices in entity_token_indices:
    entity_embeddings.append(embeddings[token_indices].mean(dim=0).tolist())

  return ResOut(entity_embeddings=entity_embeddings)


if __name__ == '__main__':
  import uvicorn
  uvicorn.run("main:app", host="localhost", port=8000, reload=True)
