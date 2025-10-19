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
import logging
import re
import math
from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
from . import query
from ..nlp import rag_tokenizer
from ...utils import rmSpace, get_float
from ...constants import TAG_FLD, PAGERANK_FLD
from app.services.common.doc_vector_store_service import DocVectorStoreService, OrderByExpr
from app.infrastructure.vector_store import (
    SearchRequest,
    MatchDenseExpr, FusionExpr, 
    SortOrder, SortFieldType, SortMode, SortField,
)


def index_name(uid): return f"{uid}"


class Dealer:
    """
    检索处理器，负责文档检索、重排序、相似度计算等功能
    """
    
    def __init__(self, dataStore: DocVectorStoreService):
        self.qryr = query.FulltextQueryer()
        self.dataStore = dataStore

    @dataclass
    class SearchResult:
        total: int
        ids: list[str]
        query_vector: list[float] | None = None
        field: dict | None = None
        highlight: dict | None = None
        aggregation: list | dict | None = None
        keywords: list[str] | None = None
        group_docs: list[list] | None = None

    async def _get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        """
        获取文本的向量表示
        
        入参:
            txt (str): 输入文本
            emb_mdl: 嵌入模型
            topk (int): 返回top-k结果，默认为10
            similarity (float): 相似度阈值，默认为0.1
            
        出参:
            MatchDenseExpr: 向量匹配表达式对象
        """
        qv, _ = await emb_mdl.encode_queries(txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(
                f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [get_float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk, {"similarity": similarity})

    def _get_filters(self, req):
        """
        从请求中提取过滤条件（私有方法）
        
        入参:
            req (dict): 搜索请求参数
            
        出参:
            dict: 过滤条件字典
        """
        condition = dict()
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            if key in req and req[key] is not None:
                condition[field] = req[key]
        # TODO(yzc): `available_int` is nullable however infinity doesn't support nullable columns.
        for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd", "removed_kwd"]:
            if key in req and req[key] is not None:
                condition[key] = req[key]
        return condition

    async def search(self, req, idx_names: str | list[str],
               kb_ids: list[str],
               emb_mdl=None,
               highlight=False,
               rank_feature: dict | None = None
               ):
        """
        执行文档搜索
        
        入参:
            req (dict): 搜索请求参数
            idx_names (str|list[str]): 索引名称
            kb_ids (list[str]): 知识库ID列表
            emb_mdl: 嵌入模型，可选
            highlight (bool): 是否高亮显示，默认为False
            rank_feature (dict): 排序特征，可选
            
        出参:
            SearchResult: 搜索结果对象
        """
        filters = self._get_filters(req)
        orderBy = OrderByExpr()

        pg = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        ps = int(req.get("size", topk))
        offset, limit = pg * ps, ps

        # 3. 设置返回字段列表
        src = req.get("fields",
                      ["docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks", "important_kwd", "position_int",
                       "doc_id", "page_num_int", "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
                       "question_kwd", "question_tks", "doc_type_kwd",
                       "available_int", "content_with_weight", PAGERANK_FLD, TAG_FLD])
        kwds = set([])

        qst = req.get("question", "")
        q_vec = []
        if not qst:
            # 5.1 无查询问题：执行基础搜索（按文档排序）
            if req.get("sort"):
                orderBy.asc("page_num_int")
                orderBy.asc("top_int")
                orderBy.desc("create_timestamp_flt")
            res = await self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
            total = self.dataStore.getTotal(res)
            logging.debug("Dealer.search TOTAL: {}".format(total))
        else:
            # 5.2 有查询问题：执行智能搜索
            highlightFields = ["content_ltks", "title_tks"] if highlight else []
            matchText, keywords = self.qryr.question(qst, min_match=0.3)  # 生成文本匹配表达式
            
            if emb_mdl is None:
                # 5.2.1 仅文本搜索：使用关键词匹配
                matchExprs = [matchText]
                res = await self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.getTotal(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))
            else:
                # 5.2.2 混合搜索：文本匹配 + 向量相似度
                matchDense = await self._get_vector(qst, emb_mdl, topk, req.get("similarity", 0.1))  # 生成向量匹配表达式
                q_vec = matchDense.embedding_data
                src.append(f"q_{len(q_vec)}_vec")

                # 5.2.3 创建融合表达式：文本权重5%，向量权重95%
                fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05, 0.95"})
                matchExprs = [matchText, matchDense, fusionExpr]

                res = await self.dataStore.search(src, highlightFields, filters, matchExprs, orderBy, offset, limit,
                                            idx_names, kb_ids, rank_feature=rank_feature)
                total = self.dataStore.getTotal(res)
                logging.debug("Dealer.search TOTAL: {}".format(total))

                # 5.2.4 搜索结果为空时的降级策略
                if total == 0:
                    if filters.get("doc_id"):
                        # 如果指定了文档ID，直接返回该文档内容
                        res = await self.dataStore.search(src, [], filters, [], orderBy, offset, limit, idx_names, kb_ids)
                        total = self.dataStore.getTotal(res)
                    else:
                        # 降低匹配阈值重试：文本匹配阈值0.1，向量相似度0.17
                        matchText, _ = self.qryr.question(qst, min_match=0.1)
                        matchDense.extra_options["similarity"] = 0.17
                        res = await self.dataStore.search(src, highlightFields, filters, [matchText, matchDense, fusionExpr],
                                                    orderBy, offset, limit, idx_names, kb_ids, rank_feature=rank_feature)
                        total = self.dataStore.getTotal(res)
                    logging.debug("Dealer.search 2 TOTAL: {}".format(total))

            # 6. 关键词扩展：将关键词进行细粒度分词，丰富关键词集合
            for k in keywords:
                kwds.add(k)
                for kk in rag_tokenizer.fine_grained_tokenize(k).split():
                    if len(kk) < 2:  # 过滤过短的词
                        continue
                    if kk in kwds:   # 避免重复
                        continue
                    kwds.add(kk)

        # 7. 构建并返回搜索结果
        logging.debug(f"TOTAL: {total}")
        ids = self.dataStore.getChunkIds(res)  # 获取文档块ID列表
        keywords = list(kwds)  # 关键词列表
        highlight = self.dataStore.getHighlight(res, keywords, "content_with_weight")  # 高亮信息
        aggs = self.dataStore.getAggregation(res, "docnm_kwd")  # 文档聚合信息
        return self.SearchResult(
            total=total,
            ids=ids,
            query_vector=q_vec,
            aggregation=aggs,
            highlight=highlight,
            field=self.dataStore.getFields(res, src),
            keywords=keywords
        )

    @staticmethod
    def _trans2floats(txt):
        """
        将字符串转换为浮点数列表（私有静态方法）
        
        入参:
            txt (str): 以制表符分隔的字符串
            
        出参:
            list[float]: 浮点数列表
        """
        return [get_float(t) for t in txt.split("\t")]

    async def insert_citations(self, answer, chunks, chunk_v,
                         embd_mdl, tkweight=0.1, vtweight=0.9):
        """
        在答案中插入引用信息
        
        入参:
            answer (str): 原始答案文本
            chunks (list): 文档块列表
            chunk_v (list): 文档块向量列表
            embd_mdl: 嵌入模型
            tkweight (float): 词汇权重，默认为0.1
            vtweight (float): 向量权重，默认为0.9
            
        出参:
            tuple: (带引用的答案, 引用ID集合)
        """
        assert len(chunks) == len(chunk_v)
        if not chunks:
            return answer, set([])
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st: i]) + "\n")
                else:
                    pieces_.extend(
                        re.split(
                            r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])",
                            pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        logging.debug("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])

        ans_v, _ = await embd_mdl.encode(pieces_)
        for i in range(len(chunk_v)):
            if len(ans_v[0]) != len(chunk_v[i]):
                chunk_v[i] = [0.0]*len(ans_v[0])
                logging.warning("The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))

        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(
            len(ans_v[0]), len(chunk_v[0]))

        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split()
                      for ck in chunks]
        cites = {}
        thr = 0.63
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i],
                                                                chunk_v,
                                                                rag_tokenizer.tokenize(
                                                                    self.qryr.rmWWW(pieces_[i])).split(),
                                                                chunks_tks,
                                                                tkweight, vtweight)
                mx = np.max(sim) * 0.99
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(
                    set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" [ID:{c}]"
                seted.add(c)

        return res, seted

    def _rank_feature_scores(self, query_rfea, search_res):
        """
        计算排序特征分数（私有方法）
        
        入参:
            query_rfea (dict): 查询排序特征
            search_res (SearchResult): 搜索结果
            
        出参:
            np.array: 排序特征分数数组
        """
        ## For rank feature(tag_fea) scores.
        rank_fea = []
        pageranks = []
        for chunk_id in search_res.ids:
            pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))
        pageranks = np.array(pageranks, dtype=float)

        if not query_rfea:
            return np.array([0 for _ in range(len(search_res.ids))]) + pageranks

        q_denor = np.sqrt(np.sum([s*s for t,s in query_rfea.items() if t != PAGERANK_FLD]))
        for i in search_res.ids:
            nor, denor = 0, 0
            if not search_res.field[i].get(TAG_FLD):
                rank_fea.append(0)
                continue
            for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc
                denor += sc * sc
            if denor == 0:
                rank_fea.append(0)
            else:
                rank_fea.append(nor/np.sqrt(denor)/q_denor)
        return np.array(rank_fea)*10. + pageranks

    def rerank(self, sres, query, tkweight=0.3,
               vtweight=0.7, cfield="content_ltks",
               rank_feature: dict | None = None
               ):
        """
        对搜索结果进行重排序
        
        入参:
            sres (SearchResult): 搜索结果
            query (str): 查询文本
            tkweight (float): 词汇权重，默认为0.3
            vtweight (float): 向量权重，默认为0.7
            cfield (str): 内容字段名，默认为"content_ltks"
            rank_feature (dict): 排序特征，可选
            
        出参:
            tuple: (相似度分数列表, 词汇相似度列表, 向量相似度列表)
        """
        _, keywords = self.qryr.question(query)
        vector_size = len(sres.query_vector)
        vector_column = f"q_{vector_size}_vec"
        zero_vector = [0.0] * vector_size
        ins_embd = []
        for chunk_id in sres.ids:
            vector = sres.field[chunk_id].get(vector_column, zero_vector)
            if isinstance(vector, str):
                vector = [get_float(v) for v in vector.split("\t")]
            ins_embd.append(vector)
        if not ins_embd:
            return [], [], []

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = list(OrderedDict.fromkeys(sres.field[i][cfield].split()))
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            question_tks = [t for t in sres.field[i].get("question_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks * 2 + important_kwd * 5 + question_tks * 6
            ins_tw.append(tks)

        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        sim, tksim, vtsim = self.qryr.hybrid_similarity(sres.query_vector,
                                                        ins_embd,
                                                        keywords,
                                                        ins_tw, tkweight, vtweight)

        return sim + rank_fea, tksim, vtsim

    async def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3,
                        vtweight=0.7, cfield="content_ltks",
                        rank_feature: dict | None = None):
        """
        使用重排序模型对搜索结果进行重排序
        
        入参:
            rerank_mdl: 重排序模型
            sres (SearchResult): 搜索结果
            query (str): 查询文本
            tkweight (float): 词汇权重，默认为0.3
            vtweight (float): 向量权重，默认为0.7
            cfield (str): 内容字段名，默认为"content_ltks"
            rank_feature (dict): 排序特征，可选
            
        出参:
            tuple: (相似度分数列表, 词汇相似度列表, 向量相似度列表)
        """
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]
        ins_tw = []
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        tksim = self.qryr.token_similarity(keywords, ins_tw)
        vtsim, _ = await rerank_mdl.similarity(query, [rmSpace(" ".join(tks)) for tks in ins_tw])
        ## For rank feature(tag_fea) scores.
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        return tkweight * (np.array(tksim)+rank_fea) + vtweight * vtsim, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        """
        计算混合相似度（私有方法）
        
        入参:
            ans_embd (list): 答案嵌入向量
            ins_embd (list): 实例嵌入向量
            ans (str): 答案文本
            inst (str): 实例文本
            
        出参:
            tuple: 混合相似度结果
        """
        return self.qryr.hybrid_similarity(ans_embd,
                                           ins_embd,
                                           rag_tokenizer.tokenize(ans).split(),
                                           rag_tokenizer.tokenize(inst).split())

    async def retrieval(self, question, embd_mdl, tenant_ids, kb_ids, page, page_size, similarity_threshold=0.2,
                  vector_similarity_weight=0.3, top=1024, doc_ids=None, aggs=True,
                  rerank_mdl=None, highlight=False,
                  rank_feature: dict | None = {PAGERANK_FLD: 10}):
        """
        执行文档检索，支持分页和重排序
        
        入参:
            question (str): 查询问题
            embd_mdl: 嵌入模型
            tenant_ids (str|list[str]): 租户ID
            kb_ids (list[str]): 知识库ID列表
            page (int): 页码
            page_size (int): 每页大小
            similarity_threshold (float): 相似度阈值，默认为0.2
            vector_similarity_weight (float): 向量相似度权重，默认为0.3
            top (int): 向量检索返回的最相似chunk数量，默认为1024
            doc_ids (list[str]): 文档ID列表，可选
            aggs (bool): 是否聚合，默认为True
            rerank_mdl: 重排序模型，可选
            highlight (bool): 是否高亮，默认为False
            rank_feature (dict): 排序特征，默认为PageRank
            
        出参:
            dict: 检索结果，包含总数、文档块列表和文档聚合信息

        执行过程说明：
            1. 向量检索：top=5，只计算5个最相似的chunks
            2. 文本检索：同时进行文本匹配检索
            3. 融合搜索：将向量检索和文本检索结果融合
            4. 数据库查询：limit=60，从融合结果中返回最多60个chunks
            5. 重排序：对这60个chunks进行重排序
            6. 最终截取：page_size=12，截取前12个chunks
        """
        # 1. 初始化返回结果结构
        ranks = {"total": 0, "chunks": [], "doc_aggs": {}}
        if not question:
            return ranks

        # 2. 计算重排序限制：优化性能，避免对大量结果进行重排序
        RERANK_LIMIT = 64  # 基础重排序限制
        # 根据页面大小调整重排序限制，确保分页对齐
        RERANK_LIMIT = int(RERANK_LIMIT//page_size + ((RERANK_LIMIT%page_size)/(page_size*1.) + 0.5)) * page_size if page_size>1 else 1
        if RERANK_LIMIT < 1:  # 当页面大小很大时，确保至少为1
            RERANK_LIMIT = 1
            
        # 3. 构建搜索请求参数
        req = {"kb_ids": kb_ids, "doc_ids": doc_ids, "page": math.ceil(page_size*page/RERANK_LIMIT), "size": RERANK_LIMIT,
               "question": question, "vector": True, "topk": top,
               "similarity": similarity_threshold,
               "available_int": 1}

        # 4. 处理租户ID格式
        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")

        sres = await self.search(req, [index_name(tid) for tid in tenant_ids],
                           kb_ids, embd_mdl, highlight, rank_feature=rank_feature)

        # 6. 重排序处理：使用重排序模型或基础重排序算法
        if rerank_mdl and sres.total > 0:
            # 6.1 使用专门的重排序模型（更精确但更慢）
            sim, tsim, vsim = await self.rerank_by_model(rerank_mdl,
                                                   sres, question, 1 - vector_similarity_weight,
                                                   vector_similarity_weight,
                                                   rank_feature=rank_feature)
        else:
            # 6.2 使用基础重排序算法（快速但相对简单）
            sim, tsim, vsim = self.rerank(
                sres, question, 1 - vector_similarity_weight, vector_similarity_weight,
                rank_feature=rank_feature)
                
        # 7. 分页处理：根据相似度分数排序并分页
        idx = np.argsort(sim * -1)[(page - 1) * page_size:page * page_size]  # 按相似度降序排序并分页
        dim = len(sres.query_vector)
        vector_column = f"q_{dim}_vec"
        zero_vector = [0.0] * dim
        
        # 8. 相似度过滤和统计
        sim_np = np.array(sim)
        if doc_ids:
            similarity_threshold = 0  # 如果指定了文档ID，不进行相似度过滤
        filtered_count = (sim_np >= similarity_threshold).sum()  # 统计满足阈值的结果数量
        ranks["total"] = int(filtered_count)  # 转换为Python int避免JSON序列化错误
        
        # 9. 构建返回结果：遍历排序后的结果，构建文档块信息
        for i in idx:
            if sim[i] < similarity_threshold:  # 相似度不满足阈值则停止
                break

            id = sres.ids[i]
            chunk = sres.field[id]
            dnm = chunk.get("docnm_kwd", "")  # 文档名称
            did = chunk.get("doc_id", "")     # 文档ID

            # 9.1 处理分页溢出：如果当前页已满，只统计聚合信息
            if len(ranks["chunks"]) >= page_size:
                if aggs:
                    if dnm not in ranks["doc_aggs"]:
                        ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
                    ranks["doc_aggs"][dnm]["count"] += 1
                    continue
                break
                
            # 9.2 构建文档块详细信息
            position_int = chunk.get("position_int", [])
            d = {
                "chunk_id": id,
                "content_ltks": chunk["content_ltks"],                    # 内容分词
                "content_with_weight": chunk["content_with_weight"],      # 带权重的内容
                "doc_id": did,
                "docnm_kwd": dnm,
                "kb_id": chunk["kb_id"],
                "important_kwd": chunk.get("important_kwd", []),          # 重要关键词
                "image_id": chunk.get("img_id", ""),                      # 图片ID
                "similarity": sim[i],                                     # 综合相似度
                "vector_similarity": vsim[i],                            # 向量相似度
                "term_similarity": tsim[i],                              # 词汇相似度
                "vector": chunk.get(vector_column, zero_vector),         # 向量表示
                "positions": position_int,                               # 位置信息
                "doc_type_kwd": chunk.get("doc_type_kwd", "")            # 文档类型
            }
            
            # 9.3 处理高亮显示
            if highlight and sres.highlight:
                if id in sres.highlight:
                    d["highlight"] = rmSpace(sres.highlight[id])
                else:
                    d["highlight"] = d["content_with_weight"]
                    
            ranks["chunks"].append(d)
            
            # 9.4 更新文档聚合统计
            if dnm not in ranks["doc_aggs"]:
                ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
            ranks["doc_aggs"][dnm]["count"] += 1
            
        # 10. 格式化文档聚合信息：按文档块数量降序排列
        ranks["doc_aggs"] = [{"doc_name": k,
                              "doc_id": v["doc_id"],
                              "count": v["count"]} for k,
                                                       v in sorted(ranks["doc_aggs"].items(),
                                                                   key=lambda x: x[1]["count"] * -1)]
        # 11. 确保返回的文档块数量不超过页面大小
        ranks["chunks"] = ranks["chunks"][:page_size]
        # 更新total为实际返回的chunks数量，保持一致性
        ranks["total"] = len(ranks["chunks"])

        return ranks

    async def sql_retrieval(self, sql, fetch_size=128, format="json"):
        """
        执行SQL查询检索
        
        入参:
            sql (str): SQL查询语句
            fetch_size (int): 获取大小，默认为128
            format (str): 返回格式，默认为"json"
            
        出参:
            list|dict: 查询结果
        """
        tbl = await self.dataStore.sql(sql, fetch_size, format)
        return tbl

    async def chunk_list(self, doc_id: str, tenant_id: str,
                   kb_ids: list[str], max_count=1024,
                   offset=0,
                   fields=["docnm_kwd", "content_with_weight", "img_id"]):
        """
        获取指定文档的块列表
        
        入参:
            doc_id (str): 文档ID
            tenant_id (str): 租户ID
            kb_ids (list[str]): 知识库ID列表
            max_count (int): 最大返回数量，默认为1024
            offset (int): 偏移量，默认为0
            fields (list[str]): 返回字段列表
            
        出参:
            list[dict]: 文档块列表
        """
        condition = {"doc_id": doc_id}
        res = []
        bs = 128
        for p in range(offset, max_count, bs):
            es_res = await self.dataStore.search(fields, [], condition, [], OrderByExpr(), p, bs, index_name(tenant_id),
                                           kb_ids)
            dict_chunks = self.dataStore.getFields(es_res, fields)
            for id, doc in dict_chunks.items():
                doc["id"] = id
            if dict_chunks:
                res.extend(dict_chunks.values())
            if len(dict_chunks.values()) < bs:
                break
        return res

    async def all_tags(self, tenant_id: str, kb_ids: list[str], S=1000):
        """
        获取所有标签信息
        
        入参:
            tenant_id (str): 租户ID
            kb_ids (list[str]): 知识库ID列表
            S (int): 平滑参数，默认为1000
            
        出参:
            list: 标签聚合结果
        """
        if not await self.dataStore.indexExist(index_name(tenant_id), kb_ids[0]):
            return []
        res = await self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        return self.dataStore.getAggregation(res, "tag_kwd")

    async def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=1000):
        """
        获取标签比例信息
        
        入参:
            tenant_id (str): 租户ID
            kb_ids (list[str]): 知识库ID列表
            S (int): 平滑参数，默认为1000
            
        出参:
            dict: 标签及其比例字典
        """
        res = await self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        res = self.dataStore.getAggregation(res, "tag_kwd")
        total = np.sum([c for _, c in res])
        return {t: (c + 1) / (total + S) for t, c in res}

    async def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        """
        为文档内容添加标签
        
        入参:
            tenant_id (str): 租户ID
            kb_ids (list[str]): 知识库ID列表
            doc (dict): 文档信息
            all_tags (dict): 所有标签信息
            topn_tags (int): 返回标签数量上限，默认为3
            keywords_topn (int): 关键词数量上限，默认为30
            S (int): 平滑参数，默认为1000
            
        出参:
            bool: 是否成功添加标签
        """
        idx_nm = index_name(tenant_id)
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []), keywords_topn)
        res = await self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return False
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        doc[TAG_FLD] = {a.replace(".", "_"): c for a, c in tag_fea if c > 0}
        return True

    async def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        """
        为查询问题添加标签
        
        入参:
            question (str): 查询问题
            tenant_ids (str|list[str]): 租户ID
            kb_ids (list[str]): 知识库ID列表
            all_tags (dict): 所有标签信息
            topn_tags (int): 返回标签数量上限，默认为3
            S (int): 平滑参数，默认为1000
            
        出参:
            dict: 查询标签字典
        """
        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]
        match_txt, _ = self.qryr.question(question, min_match=0.0)
        res = await self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        return {a.replace(".", "_"): max(1, c) for a, c in tag_fea}
