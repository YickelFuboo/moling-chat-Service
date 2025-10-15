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
import json
import logging
from collections import defaultdict
from copy import deepcopy
import json_repair
import pandas as pd
import trio
from .query_analyze_prompt import PROMPTS
from .utils import get_entity_type2sampels, get_llm_cache, set_llm_cache, get_relation
from ..utils import num_tokens_from_string, get_float, get_uuid
from ..rag.retrieval.search import Dealer, index_name
from ..llm_service import LLMType, LLMBundle
from app.services.common.doc_vector_store_service import OrderByExpr, DOC_STORE_CONN


class KGSearch(Dealer):
    """
    知识图谱搜索类，继承自Dealer，专门处理知识图谱相关的检索任务
    支持实体检索、关系检索、多跳推理、社区发现等功能
    """
    
    async def _chat(self, llm_bdl, system, history, gen_conf):
        """
        与LLM进行对话（私有方法）
        
        入参:
            llm_bdl: LLM模型包
            system (str): 系统提示词
            history (list): 对话历史
            gen_conf (dict): 生成配置
            
        出参:
            str: LLM响应文本
        """
        response = await get_llm_cache(llm_bdl.llm_name, system, history, gen_conf)
        if response:
            return response
        response = await llm_bdl.chat(system, history, gen_conf)
        if response.find("**ERROR**") >= 0:
            raise Exception(response)
        await set_llm_cache(llm_bdl.llm_name, system, response, history, gen_conf)
        return response

    async def _query_rewrite(self, llm, question, idxnms, kb_ids):
        """
        查询重写，使用LLM从问题中提取实体类型和实体名称（私有方法）
        
        入参:
            llm: 大语言模型
            question (str): 用户问题
            idxnms (list): 索引名称列表
            kb_ids (list): 知识库ID列表
            
        出参:
            tuple: (实体类型关键词列表, 从查询中提取的实体列表)
        """
        ty2ents = await get_entity_type2sampels(idxnms, kb_ids)
        hint_prompt = PROMPTS["minirag_query2kwd"].format(query=question,
                                                          TYPE_POOL=json.dumps(ty2ents, ensure_ascii=False, indent=2))
        result = await self._chat(llm, hint_prompt, [{"role": "user", "content": "Output:"}], {"temperature": .5})
        try:
            keywords_data = json_repair.loads(result)
            type_keywords = keywords_data.get("answer_type_keywords", [])
            entities_from_query = keywords_data.get("entities_from_query", [])[:5]
            return type_keywords, entities_from_query
        except json_repair.JSONDecodeError:
            try:
                result = result.replace(hint_prompt[:-1], '').replace('user', '').replace('model', '').strip()
                result = '{' + result.split('{')[1].split('}')[0] + '}'
                keywords_data = json_repair.loads(result)
                type_keywords = keywords_data.get("answer_type_keywords", [])
                entities_from_query = keywords_data.get("entities_from_query", [])[:5]
                return type_keywords, entities_from_query
            # Handle parsing error
            except Exception as e:
                logging.exception(f"JSON parsing error: {result} -> {e}")
                raise e

    def _ent_info_from_(self, es_res, sim_thr=0.3):
        """
        从ES搜索结果中提取实体信息（私有方法）
        
        入参:
            es_res: Elasticsearch搜索结果
            sim_thr (float): 相似度阈值，默认为0.3
            
        出参:
            dict: 实体信息字典，key为实体名，value包含相似度、PageRank、描述等
        """
        res = {}
        flds = ["content_with_weight", "_score", "entity_kwd", "rank_flt", "n_hop_with_weight"]
        es_res = self.dataStore.getFields(es_res, flds)
        for _, ent in es_res.items():
            for f in flds:
                if f in ent and ent[f] is None:
                    del ent[f]
            if get_float(ent.get("_score", 0)) < sim_thr:
                continue
            if isinstance(ent["entity_kwd"], list):
                ent["entity_kwd"] = ent["entity_kwd"][0]
            res[ent["entity_kwd"]] = {
                "sim": get_float(ent.get("_score", 0)),
                "pagerank": get_float(ent.get("rank_flt", 0)),
                "n_hop_ents": json.loads(ent.get("n_hop_with_weight", "[]")),
                "description": ent.get("content_with_weight", "{}")
            }
        return res

    def _relation_info_from_(self, es_res, sim_thr=0.3):
        """
        从ES搜索结果中提取关系信息（私有方法）
        
        入参:
            es_res: Elasticsearch搜索结果
            sim_thr (float): 相似度阈值，默认为0.3
            
        出参:
            dict: 关系信息字典，key为(源实体, 目标实体)元组，value包含相似度、权重、描述等
        """
        res = {}
        es_res = self.dataStore.getFields(es_res, ["content_with_weight", "_score", "from_entity_kwd", "to_entity_kwd",
                                                   "weight_int"])
        for _, ent in es_res.items():
            if get_float(ent["_score"]) < sim_thr:
                continue
            f, t = sorted([ent["from_entity_kwd"], ent["to_entity_kwd"]])
            if isinstance(f, list):
                f = f[0]
            if isinstance(t, list):
                t = t[0]
            res[(f, t)] = {
                "sim": get_float(ent["_score"]),
                "pagerank": get_float(ent.get("weight_int", 0)),
                "description": ent["content_with_weight"]
            }
        return res

    async def _get_relevant_ents_by_keywords(self, keywords, filters, idxnms, kb_ids, emb_mdl, sim_thr=0.3, N=56):
        """
        基于关键词检索相关实体（私有方法）
        
        入参:
            keywords (list): 关键词列表
            filters (dict): 过滤条件
            idxnms (list): 索引名称列表
            kb_ids (list): 知识库ID列表
            emb_mdl: 嵌入模型
            sim_thr (float): 相似度阈值，默认为0.3
            N (int): 返回结果数量上限，默认为56
            
        出参:
            dict: 相关实体信息字典
        """
        if not keywords:
            return {}
        filters = deepcopy(filters)
        filters["knowledge_graph_kwd"] = "entity"
        matchDense = self._get_vector(", ".join(keywords), emb_mdl, 1024, sim_thr)
        es_res = await self.dataStore.search(["content_with_weight", "entity_kwd", "rank_flt"], [], filters, [matchDense],
                                       OrderByExpr(), 0, N,
                                       idxnms, kb_ids)
        return self._ent_info_from_(es_res, sim_thr)

    async def _get_relevant_relations_by_txt(self, txt, filters, idxnms, kb_ids, emb_mdl, sim_thr=0.3, N=56):
        """
        基于文本检索相关关系（私有方法）
        
        入参:
            txt (str): 输入文本
            filters (dict): 过滤条件
            idxnms (list): 索引名称列表
            kb_ids (list): 知识库ID列表
            emb_mdl: 嵌入模型
            sim_thr (float): 相似度阈值，默认为0.3
            N (int): 返回结果数量上限，默认为56
            
        出参:
            dict: 相关关系信息字典
        """
        if not txt:
            return {}
        filters = deepcopy(filters)
        filters["knowledge_graph_kwd"] = "relation"
        matchDense = self._get_vector(txt, emb_mdl, 1024, sim_thr)
        es_res = await self.dataStore.search(
            ["content_with_weight", "_score", "from_entity_kwd", "to_entity_kwd", "weight_int"],
            [], filters, [matchDense], OrderByExpr(), 0, N, idxnms, kb_ids)
        return self._relation_info_from_(es_res, sim_thr)

    async def _get_relevant_ents_by_types(self, types, filters, idxnms, kb_ids, N=56):
        """
        基于实体类型检索相关实体（私有方法）
        
        入参:
            types (list): 实体类型列表
            filters (dict): 过滤条件
            idxnms (list): 索引名称列表
            kb_ids (list): 知识库ID列表
            N (int): 返回结果数量上限，默认为56
            
        出参:
            dict: 相关实体信息字典
        """
        if not types:
            return {}
        filters = deepcopy(filters)
        filters["knowledge_graph_kwd"] = "entity"
        filters["entity_type_kwd"] = types
        ordr = OrderByExpr()
        ordr.desc("rank_flt")
        es_res = await self.dataStore.search(["entity_kwd", "rank_flt"], [], filters, [], ordr, 0, N,
                                       idxnms, kb_ids)
        return self._ent_info_from_(es_res, 0)

    async def retrieval(self, question: str,
               tenant_ids: str | list[str],
               kb_ids: list[str],
               emb_mdl,
               llm,
               max_token: int = 8196,
               ent_topn: int = 6,
               rel_topn: int = 6,
               comm_topn: int = 1,
               ent_sim_threshold: float = 0.3,
               rel_sim_threshold: float = 0.3,
                  **kwargs
               ):
        """
        知识图谱检索的核心方法
        基于用户问题，从知识图谱中检索相关的实体、关系和社区信息
        
        Args:
            question: 用户问题
            tenant_ids: 租户ID列表
            kb_ids: 知识库ID列表
            emb_mdl: 嵌入模型，用于向量相似度计算
            llm: 大语言模型，用于查询重写和实体提取
            max_token: 最大token数量限制
            ent_topn: 返回的实体数量上限
            rel_topn: 返回的关系数量上限
            comm_topn: 返回的社区数量上限
            ent_sim_threshold: 实体相似度阈值
            rel_sim_threshold: 关系相似度阈值
        """
        qst = question
        # 构建检索过滤器，限制在指定的知识库范围内
        filters = self._get_filters({"kb_ids": kb_ids})
        
        # 标准化租户ID格式，确保是列表形式
        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")
        
        # 获取所有租户对应的索引名称
        idxnms = [index_name(tid) for tid in tenant_ids]
        
        # 初始化实体类型关键词列表
        ty_kwds = []
        
        try:
            # 查询重写：使用LLM从问题中提取实体类型和实体名称
            # 这是图谱检索的关键步骤，能够理解问题的语义意图
            ty_kwds, ents = await self._query_rewrite(llm, qst, [index_name(tid) for tid in tenant_ids], kb_ids)
            logging.info(f"Q: {qst}, Types: {ty_kwds}, Entities: {ents}")
        except Exception as e:
            # 如果查询重写失败，记录异常并使用原始问题作为实体
            logging.exception(e)
            ents = [qst]
            pass

        # 基于关键词的实体检索：从问题中提取的实体名称进行相似度搜索
        ents_from_query = await self._get_relevant_ents_by_keywords(ents, filters, idxnms, kb_ids, emb_mdl, ent_sim_threshold)
        
        # 基于实体类型的实体检索：根据问题中识别的实体类型进行检索
        ents_from_types = await self._get_relevant_ents_by_types(ty_kwds, filters, idxnms, kb_ids, 10000)
        
        # 基于文本的关系检索：从问题文本中检索相关的关系
        rels_from_txt = await self._get_relevant_relations_by_txt(qst, filters, idxnms, kb_ids, emb_mdl, rel_sim_threshold)
        
        # 多跳路径检索：通过实体关系网络进行N跳推理
        # 这是图谱检索的核心优势，能够发现间接相关的信息
        nhop_pathes = defaultdict(dict)
        for _, ent in ents_from_query.items():
            # 获取实体的N跳邻居信息
            nhops = ent.get("n_hop_ents", [])
            if not isinstance(nhops, list):
                logging.warning(f"Abnormal n_hop_ents: {nhops}")
                continue
            
            # 遍历每个N跳路径
            for nbr in nhops:
                path = nbr["path"]      # 路径上的实体序列
                wts = nbr["weights"]    # 路径权重
                
                # 计算路径上每对相邻实体的关系强度
                for i in range(len(path) - 1):
                    f, t = path[i], path[i + 1]  # 前一个和后一个实体
                    if (f, t) in nhop_pathes:
                        # 如果关系已存在，累加相似度（距离越远权重越低）
                        nhop_pathes[(f, t)]["sim"] += ent["sim"] / (2 + i)
                    else:
                        # 如果关系不存在，创建新的关系记录
                        nhop_pathes[(f, t)]["sim"] = ent["sim"] / (2 + i)
                    nhop_pathes[(f, t)]["pagerank"] = wts[i]

        # 记录检索结果的统计信息
        logging.info("Retrieved entities: {}".format(list(ents_from_query.keys())))
        logging.info("Retrieved relations: {}".format(list(rels_from_txt.keys())))
        logging.info("Retrieved entities from types({}): {}".format(ty_kwds, list(ents_from_types.keys())))
        logging.info("Retrieved N-hops: {}".format(list(nhop_pathes.keys())))

        # 实体相似度增强：如果实体既来自查询又来自类型，则增强其相似度
        # P(E|Q) = P(E) * P(Q|E) => pagerank * similarity
        for ent in ents_from_types.keys():
            if ent not in ents_from_query:
                continue
            # 双重匹配的实体获得2倍相似度提升
            ents_from_query[ent]["sim"] *= 2

        # 关系相似度增强：综合考虑多种来源的关系信息
        for (f, t) in rels_from_txt.keys():
            pair = tuple(sorted([f, t]))  # 标准化关系对（无向图）
            s = 0
            
            # 如果关系在多跳路径中存在，累加其相似度
            if pair in nhop_pathes:
                s += nhop_pathes[pair]["sim"]
                del nhop_pathes[pair]  # 避免重复计算
            
            # 如果关系的端点实体在类型匹配中，增加权重
            if f in ents_from_types:
                s += 1
            if t in ents_from_types:
                s += 1
            
            # 最终关系相似度 = 原始相似度 * (多跳权重 + 类型匹配权重 + 1)
            rels_from_txt[(f, t)]["sim"] *= s + 1

        # 处理仅来自多跳路径的关系（不在文本检索中）
        for (f, t) in nhop_pathes.keys():
            s = 0
            # 计算端点实体的类型匹配权重
            if f in ents_from_types:
                s += 1
            if t in ents_from_types:
                s += 1
            
            # 将多跳路径中的关系添加到关系列表中
            rels_from_txt[(f, t)] = {
                "sim": nhop_pathes[(f, t)]["sim"] * (s + 1),
                "pagerank": nhop_pathes[(f, t)]["pagerank"]
            }

        # 实体排序：按相似度 * PageRank的乘积排序，取前ent_topn个
        ents_from_query = sorted(ents_from_query.items(), key=lambda x: x[1]["sim"] * x[1]["pagerank"], reverse=True)[
                          :ent_topn]
        
        # 关系排序：按相似度 * PageRank的乘积排序，取前rel_topn个
        rels_from_txt = sorted(rels_from_txt.items(), key=lambda x: x[1]["sim"] * x[1]["pagerank"], reverse=True)[
                        :rel_topn]

        # 构建实体结果列表，同时控制token数量
        ents = []
        relas = []
        
        # 处理实体结果
        for n, ent in ents_from_query:
            ents.append({
                "Entity": n,  # 实体名称
                "Score": "%.2f" % (ent["sim"] * ent["pagerank"]),  # 综合评分
                "Description": json.loads(ent["description"]).get("description", "") if ent["description"] else ""  # 实体描述
            })
            # 更新剩余token数量
            max_token -= num_tokens_from_string(str(ents[-1]))
            if max_token <= 0:
                ents = ents[:-1]  # 如果超出token限制，移除最后一个实体
                break

        # 处理关系结果
        for (f, t), rel in rels_from_txt:
            # 如果关系没有描述，尝试从图谱中获取
            if not rel.get("description"):
                for tid in tenant_ids:
                    rela = get_relation(tid, kb_ids, f, t)
                    if rela:
                        break
                else:
                    continue
                rel["description"] = rela["description"]
            
            # 解析关系描述
            desc = rel["description"]
            try:
                desc = json.loads(desc).get("description", "")
            except Exception:
                pass
            
            relas.append({
                "From Entity": f,  # 源实体
                "To Entity": t,    # 目标实体
                "Score": "%.2f" % (rel["sim"] * rel["pagerank"]),  # 综合评分
                "Description": desc  # 关系描述
            })
            
            # 更新剩余token数量
            max_token -= num_tokens_from_string(str(relas[-1]))
            if max_token <= 0:
                relas = relas[:-1]  # 如果超出token限制，移除最后一个关系
                break

        # 格式化实体结果：转换为CSV表格格式
        if ents:
            ents = "\n---- Entities ----\n{}".format(pd.DataFrame(ents).to_csv())
        else:
            ents = ""
        
        # 格式化关系结果：转换为CSV表格格式
        if relas:
            relas = "\n---- Relations ----\n{}".format(pd.DataFrame(relas).to_csv())
        else:
            relas = ""

        # 返回标准化的chunk格式，包含实体、关系和社区信息
        return {
                "chunk_id": get_uuid(),  # 生成唯一ID
                "content_ltks": "",      # 空的分词内容
                "content_with_weight": ents + relas + await self._community_retrival_([n for n, _ in ents_from_query], filters, kb_ids, idxnms,
                                                        comm_topn, max_token),  # 组合所有内容
                "doc_id": "",            # 空文档ID
                "docnm_kwd": "Related content in Knowledge Graph",  # 标识为图谱内容
                "kb_id": kb_ids,         # 知识库ID列表
                "important_kwd": [],     # 空关键词列表
                "image_id": "",          # 空图片ID
                "similarity": 1.,        # 最大相似度
                "vector_similarity": 1., # 最大向量相似度
                "term_similarity": 0,    # 零术语相似度
                "vector": [],            # 空向量
                "positions": [],         # 空位置信息
            }

    async def _community_retrival_(self, entities, condition, kb_ids, idxnms, topn, max_token):
        """
        社区检索，获取与实体相关的社区报告（私有方法）
        
        入参:
            entities (list): 实体列表
            condition (dict): 检索条件
            kb_ids (list): 知识库ID列表
            idxnms (list): 索引名称列表
            topn (int): 返回结果数量上限
            max_token (int): 最大token数量限制
            
        出参:
            str: 格式化的社区报告文本
        """
        ## Community retrieval
        fields = ["docnm_kwd", "content_with_weight"]
        odr = OrderByExpr()
        odr.desc("weight_flt")
        fltr = deepcopy(condition)
        fltr["knowledge_graph_kwd"] = "community_report"
        fltr["entities_kwd"] = entities
        comm_res = await self.dataStore.search(fields, [], fltr, [],
                                         OrderByExpr(), 0, topn, idxnms, kb_ids)
        comm_res_fields = self.dataStore.getFields(comm_res, fields)
        txts = []
        for ii, (_, row) in enumerate(comm_res_fields.items()):
            obj = json.loads(row["content_with_weight"])
            txts.append("# {}. {}\n## Content\n{}\n## Evidences\n{}\n".format(
                ii + 1, row["docnm_kwd"], obj["report"], obj["evidences"]))
            max_token -= num_tokens_from_string(str(txts[-1]))

        if not txts:
            return ""
        return "\n---- Community Report ----\n" + "\n".join(txts)


if __name__ == "__main__":
    import argparse
    from ..rag.retrieval import search
    from app.services.kb_service import KBService



    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tenant_id', default=False, help="Tenant ID", action='store', required=True)
    parser.add_argument('-d', '--kb_id', default=False, help="Knowledge base ID", action='store', required=True)
    parser.add_argument('-q', '--question', default=False, help="Question", action='store', required=True)
    args = parser.parse_args()

    kb_id = args.kb_id
    kb = KBService.get_kb_by_id(kb_id)
    tenant_id = kb.tenant_id
    llm_bdl = LLMBundle(tenant_id, LLMType.CHAT)
    embed_bdl = LLMBundle(tenant_id, LLMType.EMBEDDING)
    
    kg = KGSearch(DOC_STORE_CONN)
    print(kg.retrieval({"question": args.question, "kb_ids": [kb_id]},
                    search.index_name(kb.tenant_id), [kb_id], embed_bdl, llm_bdl))
