"""
title: DataDog Filter Pipeline
author: 0xThresh
date: 2024-06-06
version: 1.0
license: MIT
description: A filter pipeline that sends traces to DataDog.
requirements: ddtrace
environment_variables: DD_LLMOBS_AGENTLESS_ENABLED, DD_LLMOBS_ENABLED, DD_LLMOBS_APP_NAME, DD_API_KEY, DD_SITE 
"""

import json
from typing import List, Optional
import os
import numpy as np
from utils.pipelines.main import get_last_user_message, get_last_assistant_message
from pydantic import BaseModel
from PIL import Image
from io import BytesIO
import base64
from transformers import AutoModel, AutoImageProcessor
import torch
import torch.nn.functional as F
import faiss
import pandas as pd

RAG_PROMPT = """Use the following context to answer the user query:\n{context}"""

def get_rag_prompt(context: List[str]):
    return RAG_PROMPT.format(context='\n\n##\n'.join(context))

def is_debug():
    return int(os.getenv("DEBUG", 0))

def load_config():
    with open("pipelines/config.json", "r") as f:
        return json.load(f)

class Retriever:
    def __init__(self, dict_info):
        print("Loading retrieval index")
        self.index = faiss.read_index(os.path.join(dict_info['index_path'], 'knn.index'))
        self.values = json.load(open(os.path.join(dict_info['index_path'], 'knn.json'), 'r'))
        print("Done loading retrieval index")

    def retrieve(self, query, k):
        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()
        query = query.astype(np.float32)
        D, indexes = self.index.search(query, k=k)
        chosen_k = indexes[0, :k].tolist()
        return [self.values[k][0] for k in chosen_k]


class Pipeline:
    class Valves(BaseModel):
        # List target pipeline ids (models) that this filter will be connected to.
        # If you want to connect this filter to all pipelines, you can set pipelines to ["*"]
        # e.g. ["llama3:latest", "gpt-3.5-turbo"]
        pipelines: List[str] = []

        # Assign a priority level to the filter pipeline.
        # The priority level determines the order in which the filter pipelines are executed.
        # The lower the number, the higher the priority.
        priority: int = 0

        # Valves
        dd_api_key: str
        dd_site: str
        ml_app: str

    def __init__(self):
        # Pipeline filters are only compatible with Open WebUI
        # You can think of filter pipeline as a middleware that can be used to edit the form data before it is sent to the OpenAI API.
        self.type = "filter"

        # Optionally, you can set the id and name of the pipeline.
        # Best practice is to not specify the id so that it can be automatically inferred from the filename, so that users can install multiple versions of the same pipeline.
        # The identifier must be unique across all pipelines.
        # The identifier must be an alphanumeric string that can include underscores or hyphens. It cannot contain spaces, special characters, slashes, or backslashes.
        # self.id = "datadog_filter_pipeline"
        self.name = "DataDog Filter"

        # Initialize
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],  # Connect to all pipelines
                "dd_api_key": os.getenv("DD_API_KEY", "foobar"),
                "dd_site": os.getenv("DD_SITE", "datadoghq.com"),
                "ml_app": os.getenv("ML_APP", "pipelines-test"),
            }
        )

        # DataDog LLMOBS docs: https://docs.datadoghq.com/tracing/llm_observability/sdk/
        self.llm_span = None
        self.chat_generations = {}

        self.config = load_config()
        
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'
        self.image_encoder = AutoModel.from_pretrained(self.config['model_name_or_path']).to(self.device)
        self.image_processor = AutoImageProcessor.from_pretrained(self.config['model_name_or_path'])

        self.retriever = Retriever(self.config)

        self.kb = pd.read_json(self.config['kb_jsonl_path'], lines=True, nrows=10000 if is_debug() else None)
        print(f"Loaded {len(self.kb)} entities from {self.config['kb_jsonl_path']}")
        self.entity_k = self.config['entity_k']

    async def on_startup(self):
        # This function is called when the server is started.
        print(f"on_startup:{__name__}")
        self.set_dd()
        pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        print(f"on_shutdown:{__name__}")
        self.LLMObs.flush()
        pass

    async def on_valves_updated(self):
        # This function is called when the valves are updated.
        self.set_dd()
        pass

    def set_dd(self):
        pass
        # self.LLMObs.enable(
        #     ml_app=self.valves.ml_app,
        #     api_key=self.valves.dd_api_key,
        #     site=self.valves.dd_site,
        #     agentless_enabled=True,
        #     integrations_enabled=True,
        # )

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"inlet:{__name__}")

        # self.llm_span = self.LLMObs.llm(
        #     model_name=body["model"],
        #     name=f"filter:{__name__}",
        #     model_provider="open-webui",
        #     session_id=body["chat_id"],
        #     ml_app=self.valves.ml_app
        # )
        if isinstance(body['messages'][-1]['content'], list):
            rag_prompt = ''
            for i, el in enumerate(body['messages'][-1]['content']):
                # if el['type'] == 'text':
                #     body['messages'][-1]['content'][i]['text'] += ' Rispondi in italiano'
                if el['type'] == 'image_url':
                    img = body['messages'][-1]['content'][1]['image_url']['url']
                    img = Image.open(BytesIO(base64.b64decode(img.split('base64,')[-1])))
                    img = self.image_processor([img], return_tensors='pt').to(self.device)
                    with torch.inference_mode():
                        image_features = self.image_encoder.get_image_features(**img)
                    image_features = F.normalize(image_features, p=2, dim=-1)
                    retr_ids = self.retriever.retrieve(image_features, k=self.entity_k)
                    if is_debug():
                        retr_ids = self.kb.url.sample(self.entity_k).to_list()
                    for retr_id in retr_ids[:1]:
                        entity = self.kb[self.kb.url == retr_id]
                        if entity.empty:
                            continue
                        entity = entity.iloc[0]
                        context = entity.section_texts
                        if not isinstance(context, list):
                            context = [context]
                        rag_prompt += get_rag_prompt(context)
            if rag_prompt:
                for i, el in enumerate(body['messages'][-1]['content']):
                    if el['type'] == 'text':
                        body['messages'][-1]['content'][i]['text'] += f"\n{rag_prompt}"

        else:
            body['messages'][-1]['content'] += ' Rispondi in italiano'

        return body


    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        print(f"outlet:{__name__}")

        # self.LLMObs.annotate(
        #     span = self.llm_span,
        #     output_data = get_last_assistant_message(body["messages"]),
        # )

        # self.llm_span.finish()
        # self.LLMObs.flush()

        return body
