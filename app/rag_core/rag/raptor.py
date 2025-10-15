#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import asyncio
import logging
import re
import umap
import numpy as np
from sklearn.mixture import GaussianMixture
from ..graphrag.utils import (
    get_llm_cache,
    get_embed_cache,
    set_embed_cache,
    set_llm_cache,
)
from ..constants import CHAT_LIMITER
from ..utils import truncate, timeout


class RecursiveAbstractiveProcessing4TreeOrganizedRetrieval:
    def __init__(
        self, max_cluster, llm_model, embd_model, prompt, max_token=512, threshold=0.1
    ):
        self._max_cluster = max_cluster
        self._llm_model = llm_model
        self._embd_model = embd_model
        self._threshold = threshold
        self._prompt = prompt
        self._max_token = max_token

    @timeout(60)
    async def _chat(self, system, history, gen_conf):
        response = await get_llm_cache(self._llm_model.llm_name, system, history, gen_conf)
        if response:
            return response
        response = await self._llm_model.chat(system, history, gen_conf)
        response = re.sub(r"^.*</think>", "", response, flags=re.DOTALL)
        if response.find("**ERROR**") >= 0:
            raise Exception(response)
        await set_llm_cache(self._llm_model.llm_name, system, response, history, gen_conf)
        return response

    @timeout(2)
    async def _embedding_encode(self, txt):
        response = await get_embed_cache(self._embd_model.llm_name, txt)
        if response is not None:
            return response
        embds, _ = await self._embd_model.encode([txt])
        if len(embds) < 1 or len(embds[0]) < 1:
            raise Exception("Embedding error: ")
        embds = embds[0]
        await set_embed_cache(self._embd_model.llm_name, txt, embds)
        return embds

    def _get_optimal_clusters(self, embeddings: np.ndarray, random_state: int):
        max_clusters = min(self._max_cluster, len(embeddings))
        n_clusters = np.arange(1, max_clusters)
        bics = []
        for n in n_clusters:
            gm = GaussianMixture(n_components=n, random_state=random_state)
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        optimal_clusters = n_clusters[np.argmin(bics)]
        return optimal_clusters

    @timeout(60)
    async def _summarize_cluster(self, ck_idx: list[int], chunks: list) -> tuple:
        """
        总结一个聚类的内容
        
        Args:
            ck_idx: 聚类中chunk的索引列表
            chunks: chunks列表
            
        Returns:
            tuple: (summary_content, embedding_vector)
        """

        # 获取指定chunk单元的内容，并合并为\n分割的文本
        texts = [chunks[i][0] for i in ck_idx]
        len_per_chunk = int(
            (self._llm_model.max_length - self._max_token) / len(texts)
        )
        cluster_content = "\n".join(
            [truncate(t, max(1, len_per_chunk)) for t in texts]
        )
        
        # 新增：检查prompt是否有效
        if not self._prompt:
            logging.warning("RAPTOR prompt is empty, using default prompt")
            prompt_content = f"请总结以下段落。 小心数字，不要编造。 段落如下：\n{cluster_content}\n以上就是你需要总结的内容。"
        else:
            try:
                prompt_content = self._prompt.format(
                    cluster_content=cluster_content
                )
            except (KeyError, ValueError) as e:
                logging.warning(f"RAPTOR prompt format error: {e}, using default prompt")
                prompt_content = f"请总结以下段落。 小心数字，不要编造。 段落如下：\n{cluster_content}\n以上就是你需要总结的内容。"
        
        # 使用大语言模型进行总结
        async with CHAT_LIMITER:
            cnt = await self._chat(
                "You're a helpful assistant.",
                [
                    {
                        "role": "user",
                        "content": prompt_content,
                    }
                ],
                {"temperature": 0.3, "max_tokens": self._max_token},
            )
            cnt = re.sub(
                "(······\n由于长度的原因，回答被截断了，要继续吗？|For the content length reason, it stopped, continue?)",
                "",
                cnt,
            )
            logging.debug(f"SUM: {cnt}")

            # 总结结果计算向量
            embds = await self._embedding_encode(cnt)
            return cnt, embds

    async def __call__(self, chunks, random_state, callback=None):
        """
        RAPTOR算法的核心方法：递归聚类和总结
        
        Args:
            chunks: 待处理的文本块列表，每个元素为(text, embedding)元组
            random_state: 随机种子，确保结果可重现
            callback: 进度回调函数
            
        Returns:
            list: 经过RAPTOR处理后的chunks列表
        """
        # 如果chunks数量<=1，不聚类需要总结
        if len(chunks) <= 1:
            return []
        
        # 过滤掉空文本或空向量的chunks
        chunks = [(s, a) for s, a in chunks if s and len(a) > 0]
        
        # 初始化层级记录和索引范围
        layers = [(0, len(chunks))]  # 记录每层的chunks范围
        start, end = 0, len(chunks)  # 当前处理范围的起始和结束索引
        labels = []  # 记录每层的聚类标签
        

        while end - start > 1:
            # 获取待处理的chunks中的向量信息集
            embeddings = [embd for _, embd in chunks[start:end]]
            
            # 特殊情况：只有2个chunks时，直接合并
            if len(embeddings) == 2:
                cnt, embds = await self._summarize_cluster([start, start + 1], chunks)
                chunks.append((cnt, embds))
                if callback:
                    callback(
                        msg="Cluster one layer: {} -> {}".format(
                            end - start, len(chunks) - end
                        )
                    )
                labels.extend([0, 0])
                layers.append((end, len(chunks)))
                start = end
                end = len(chunks)
                continue

            # 计算UMAP降维参数
            n_neighbors = int((len(embeddings) - 1) ** 0.8)
            
            # 使用UMAP进行降维，减少计算复杂度
            reduced_embeddings = umap.UMAP(
                n_neighbors=max(2, n_neighbors),
                n_components=min(12, len(embeddings) - 2),
                metric="cosine",
            ).fit_transform(embeddings)
            
            # 确定最优聚类数量
            n_clusters = self._get_optimal_clusters(reduced_embeddings, random_state)
            
            # 根据聚类数量进行不同的处理
            if n_clusters == 1:
                # 只有一个聚类，所有chunks标记为同一类
                lbls = [0 for _ in range(len(reduced_embeddings))]
            else:
                # 使用高斯混合模型进行聚类
                gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
                gm.fit(reduced_embeddings)
                probs = gm.predict_proba(reduced_embeddings)
                
                # 根据概率阈值确定聚类标签
                lbls = [np.where(prob > self._threshold)[0] for prob in probs]
                lbls = [lbl[0] if isinstance(lbl, np.ndarray) else lbl for lbl in lbls]

            # 并行处理所有聚类：每个聚类独立总结
            tasks = []
            for c in range(n_clusters):
                # 找到属于当前聚类的chunk索引
                ck_idx = [i + start for i in range(len(lbls)) if lbls[i] == c]
                assert len(ck_idx) > 0
                tasks.append(self._summarize_cluster(ck_idx, chunks))
            
            # 并行执行所有聚类的总结任务
            results = await asyncio.gather(*tasks, return_exceptions=True)
            # 检查是否有异常并处理结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"RAPTOR summarize cluster {i} failed: {result}")
                    raise result
                # 添加成功的结果到chunks
                cnt, embds = result
                chunks.append((cnt, embds))

            # 验证结果：新生成的chunks数量应该等于聚类数量
            assert len(chunks) - end == n_clusters, "{} vs. {}".format(
                len(chunks) - end, n_clusters
            )
            
            # 更新标签和层级信息
            labels.extend(lbls)
            layers.append((end, len(chunks)))
            if callback:
                callback(
                    msg="Cluster one layer: {} -> {}".format(
                        end - start, len(chunks) - end
                    )
                )
            start = end
            end = len(chunks)

        return chunks
