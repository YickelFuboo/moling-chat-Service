import asyncio
from typing import Optional
from enum import StrEnum
from app.infrastructure.llms import llm_factory, cv_factory, tts_factory, embedding_factory, rerank_factory, stt_factory


class LLMType(StrEnum):
    """LLM模型类型枚举"""
    CHAT = 'chat'
    EMBEDDING = 'embedding'
    SPEECH2TEXT = 'speech2text'
    IMAGE2TEXT = 'image2text'
    RERANK = 'rerank'
    TTS = 'tts'

class LLMBundle:
    def __init__(self, tenant_id: str, llm_type: LLMType, provider: Optional[str] = None, model: Optional[str] = None, llm_name=None, lang="Chinese"):
        self.tenant_id = tenant_id
        self.llm_type = llm_type
        if llm_type == LLMType.CHAT:
            self.mdl = llm_factory.create_model(provider=provider, model=model, language=lang)
        elif llm_type == LLMType.IMAGE2TEXT:
            self.mdl = cv_factory.create_model(provider=provider, model=model, language=lang)
        elif llm_type == LLMType.TTS:
            self.mdl = tts_factory.create_model(provider=provider, model=model)
        elif llm_type == LLMType.EMBEDDING:
            self.mdl = embedding_factory.create_model(provider=provider, model=model)
        elif llm_type == LLMType.RERANK:
            self.mdl = rerank_factory.create_model(provider=provider, model=model)
        elif llm_type == LLMType.SPEECH2TEXT:
            self.mdl = stt_factory.create_model(provider=provider, model=model)
        else:
            raise ValueError(f"Unsupported model type: {llm_type.value}")
        
        self.llm_name = self.mdl.model_name
        self.max_length = self.mdl.max_length if hasattr(self.mdl, 'max_length') else 8192

    async def encode(self, texts: list):
        embeddings, used_tokens = await self.mdl.encode(texts)
        return embeddings, used_tokens

    async def encode_queries(self, query: str):
        emd, used_tokens = await self.mdl.encode_queries(query)
        return emd, used_tokens

    async def similarity(self, query: str, texts: list):
        sim, used_tokens = await self.mdl.similarity(query, texts)
        return sim, used_tokens

    async def describe(self, image, max_tokens=300):
        txt, used_tokens = await self.mdl.describe(image)
        return txt

    async def describe_with_prompt(self, image, prompt):
        txt, used_tokens = await self.mdl.describe_with_prompt(image, prompt)
        return txt

    async def transcription(self, audio):
        txt, used_tokens = await self.mdl.stt(audio)
        return txt

    async def tts(self, text):    
        async for chunk in self.mdl.tts(text):
            yield chunk

    def _remove_reasoning_content(self, txt: str) -> str:
        first_think_start = txt.find("<think>")
        if first_think_start == -1:
            return txt

        last_think_end = txt.rfind("</think>")
        if last_think_end == -1:
            return txt

        if last_think_end < first_think_start:
            return txt

        return txt[last_think_end + len("</think>") :]

    async def chat(self, system, history, gen_conf):  
        # 根据模型类型调用不同的chat接口
        if self.llm_type == LLMType.CHAT:
            # Chat模型的接口：system_prompt, user_prompt, user_question, history
            response, used_tokens = await self.mdl.chat(
                system_prompt=system, 
                user_prompt=None, 
                user_question=None, 
                history=history, 
                **gen_conf)
            # 如果返回的是ChatResponse对象，提取content属性
            if hasattr(response, 'content'):
                txt = response.content
            else:
                txt = response
        else:
            raise ValueError(f"不支持的模型类型进行chat操作: {self.llm_type}")
        
        txt = self._remove_reasoning_content(txt)
        return txt

    async def chat_stream(self, system, history, gen_conf):
        # 根据模型类型调用不同的chat_stream接口
        if self.llm_type == LLMType.CHAT:
            # Chat模型的接口：system_prompt, user_prompt, user_question, history
            stream_generator, _ = await self.mdl.chat_stream(
                system_prompt=system, 
                user_prompt=None, 
                user_question=None, 
                history=history, 
                **gen_conf)
            async for txt in stream_generator:
                yield txt
        else:
            raise ValueError(f"不支持的模型类型进行chat_stream操作: {self.llm_type}")

    async def is_strong_enough(self):
        if self.llm_type == LLMType.CHAT or self.llm_type == LLMType.EMBEDDING:
            return await self.mdl.is_strong_enough()
        else:
            return True 

    async def get_embedding_vector_size(self):
        if self.llm_type == LLMType.EMBEDDING:
            return await self.mdl.get_embedding_vector_size()
        else:
            return 0