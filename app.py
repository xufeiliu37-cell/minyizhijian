import os
import re
import json
import warnings

warnings.filterwarnings("ignore")

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatZhipuAI


# =========================================================
# 页面配置
# =========================================================
st.set_page_config(
    page_title="闽遗智鉴",
    page_icon="🏮",
    layout="wide"
)

# =========================================================
# 页面样式
# =========================================================
st.markdown("""
<style>
.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #8B2323;
    margin-bottom: 5px;
}
.sub-title {
    font-size:18px;
    color:#555555;
    margin-bottom:25px;
}
.result-card {
    padding: 18px;
    border-radius: 14px;
    background-color: #ffffff;
    border: 1px solid #eeeeee;
    margin-bottom: 18px;
}
.fact-text {
    font-size: 17px;
    font-weight: 600;
    line-height: 1.7;
}
.small-text {
    color: #666666;
    line-height: 1.6;
}
.stat-number {
    font-size: 34px;
    font-weight: 700;
}
.evidence-box {
    background-color: #f7f9fc;
    padding: 15px;
    border-radius: 10px;
    border-left: 4px solid #4CAF50;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 标题
# =========================================================
st.markdown(
    '<div class="main-title">🏮 闽遗智鉴</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">福建非遗 AIGC 内容事实核验与权威证据溯源系统</div>',
    unsafe_allow_html=True
)

# =========================================================
# 侧边栏
# =========================================================
with st.sidebar:
    st.header("🏮 系统说明")
    st.markdown("""
### 系统功能
本系统面向福建非物质文化遗产相关 AIGC 内容，利用：
- 🤖 大语言模型
- 🔎 RAG 检索增强生成
- 📚 福建非遗知识库
- 🧠 向量相似度检索
- 🔗 权威证据溯源

对输入文本进行事实核验。

---
### 核验状态
🟢 **有据**   知识库存在较明确的证据支持。

🔴 **疑误**   输入事实与知识库中的权威信息存在明显不一致。

🟡 **存疑**   存在一定相关证据，但不足以完全确认。

⚪ **无据**   当前知识库中没有找到足够支持证据。

---
### 数据来源
当前知识库主要整理福建非遗相关权威资料，并保留：
- 项目名称
- 权威来源
- 原始证据
- 来源链接
""")

# =========================================================
# API Key 直接写死在这里
# =========================================================
api_key = "ebbac0269b9c4b9ab5a767c3c9a9aaca.LTBQ3xihrYpRTHqz"

if not api_key:
    st.warning("API密钥为空，请检查代码内密钥。")


# =========================================================
# 加载 Embedding
# =========================================================
@st.cache_resource
def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


# =========================================================
# 加载向量库 Chroma
# =========================================================
@st.cache_resource
def load_database():
    embedding = load_embedding()
    db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    )
    return db


# =========================================================
# 加载大模型
# =========================================================
@st.cache_resource
def load_llm():
    return ChatZhipuAI(
        model="glm-4-flash",
        api_key=api_key,
        temperature=0
    )


# =========================================================
# 文本事实拆分
# =========================================================
def split_fact_sentences(text):
    """
    将输入文本拆成可以独立核验的事实。
    关键要求：
    1. 每条事实必须保留完整的项目名称/主语
    2. 不允许出现“包括号竹”“手绘”“上油”这种孤立短语
    3. 不改变原文事实含义
    4. 不增加原文没有的信息
    """

    llm = load_llm()

    prompt = f"""
你是一名严谨的福建非物质文化遗产事实核验助手。

请将下面这段文本拆分成若干条可以独立进行知识库检索和事实核验的事实。

【原文】
{text}

【拆分要求】
1. 每条必须是一条完整、可独立理解的事实。
2. 每条事实都必须保留对应的“项目名称/主体”。
3. 如果原文是“福州油纸伞制作工序包括号竹、制伞骨、上伞面、手绘、上油、包头”，
   不能拆成：
   “包括号竹”
   “制伞骨”
   “手绘”
   “上油”
   “包头”
4. 正确的拆分方式应该类似：
   “福州油纸伞制作工序包括号竹”
   “福州油纸伞制作工序包括制伞骨”
   “福州油纸伞制作工序包括上伞面”
   “福州油纸伞制作工序包括手绘”
   “福州油纸伞制作工序包括上油”
   “福州油纸伞制作工序包括包头”
5. 如果多条内容共享同一个项目名称，必须在每一条中重复写出项目名称。
6. 不要补充原文不存在的信息。
7. 不要修改原文的年份、地点、项目名称、工序名称等。
8. 最多拆成6条。
9. 只输出JSON数组，不要输出任何解释文字。

输出格式：
[
  "事实1",
  "事实2"
]

【待拆分文本】
{text}
"""

    try:
        resp = llm.invoke(prompt)

        content = resp.content.strip()

        # 去掉可能出现的 Markdown 代码块
        content = re.sub(r"```json|```", "", content).strip()

        facts = json.loads(content)

        # 确保返回的是列表
        if not isinstance(facts, list):
            return [text]

        # 清理空内容
        facts = [
            str(x).strip()
            for x in facts
            if str(x).strip()
        ]

        return facts[:6]

    except Exception as e:
        print(f"事实拆分失败：{e}")
        return [text]
    llm = load_llm()
    prompt = f"""
你是一名严谨的福建非物质文化遗产事实核验助手。
请将下面这段文本拆分成若干条可以独立核验的事实陈述。
要求：
1.每条为独立事实，不修改原文语义
2.不增加原文不存在信息
3.最多6条
4.只输出JSON数组，不要其他输出

示例输出：
["事实1","事实2"]

待处理文本：
{text}
"""
    try:
        resp = llm.invoke(prompt)
        content = re.sub(r"```json|```", "", resp.content.strip())
        data = json.loads(content)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
    except Exception:
        pass
    #兜底按句号分割
    parts = re.split(r"[。！？]", text)
    return [p.strip() for p in parts if len(p.strip())>6][:6]


# =========================================================
# 证据去重
# =========================================================
def deduplicate_evidence(evidence_list):
    seen = set()
    out = []
    for item in evidence_list:
        c = item.get("content","").strip()
        if c and c not in seen:
            seen.add(c)
            out.append(item)
    return out


def infer_project_name(content):
    text = re.sub(r"\s+","",content.strip())
    pat = r"^(.{2,20}?)(原名|又名|属于|是|为|被称为)"
    m = re.search(pat,text)
    if m:
        return m.group(1).strip("，。；、")
    return "福建非遗"


def load_all_documents():
    db = load_database()
    raw = db.get(include=["documents","metadatas"])
    docs = raw.get("documents",[])
    metas = raw.get("metadatas",[])
    res = []
    for idx,doc in enumerate(docs):
        meta = metas[idx] if idx<len(metas) and metas[idx] else {}
        proj = infer_project_name(doc) or meta.get("project_name","福建非遗")
        res.append({
            "content":doc.strip(),
            "source":meta.get("source","未知来源"),
            "url":meta.get("url",""),
            "project_name":proj
        })
    return res


# =========================================================
# 获取证据
# =========================================================
def get_evidence(fact, threshold):
    """
    混合证据检索：
    1. 向量语义检索
    2. 知识库全文关键词/实体匹配
    3. 自动识别项目名称
    4. 优先返回同一项目的证据
    5. 防止“莆仙戏”检索出“福州油纸伞”这种串证
    """

    import re

    db = load_database()

    fact = str(fact).strip()
    if not fact:
        return []

    # =========================================================
    # 第一部分：向量检索
    # =========================================================

    vector_results = []

    try:
        vector_results = db.similarity_search_with_relevance_scores(
            fact,
            k=20
        )
    except Exception:
        vector_results = []

    # =========================================================
    # 第二部分：读取知识库全部文档
    # 用于关键词/项目名称兜底
    # =========================================================

    all_docs = []

    try:
        data = db._collection.get(
            include=["documents", "metadatas"]
        )

        documents = data.get("documents", []) or []
        metadatas = data.get("metadatas", []) or []

        for i, content in enumerate(documents):

            if not content:
                continue

            metadata = {}

            if i < len(metadatas) and metadatas[i]:
                metadata = metadatas[i]

            all_docs.append({
                "content": str(content),
                "metadata": metadata
            })

    except Exception:
        # 如果无法读取全部文档，就至少使用向量检索结果
        all_docs = []

    # =========================================================
    # 第三部分：建立项目名称集合
    # =========================================================

    project_names = set()

    for item in all_docs:

        metadata = item["metadata"]

        project_name = metadata.get("project_name", "")

        if project_name:
            project_name = str(project_name).strip()

            if (
                project_name
                and project_name != "福建非遗"
                and len(project_name) >= 2
            ):
                project_names.add(project_name)

    # =========================================================
    # 第四部分：判断当前事实属于哪个项目
    # =========================================================

    matched_projects = []

    # 4.1 直接匹配项目名称
    for project_name in project_names:

        if project_name in fact:
            matched_projects.append(project_name)

    # =========================================================
    # 4.2 如果事实没有直接出现项目名，
    #     根据事实中的关键词寻找对应项目
    #
    #     例如：
    #     “原名兴化戏”
    #
    #     虽然没有写“莆仙戏”，
    #     但知识库中“莆仙戏”的资料包含“兴化戏”
    # =========================================================

    if not matched_projects and all_docs:

        # 提取中文连续片段
        chinese_parts = re.findall(
            r'[\u4e00-\u9fff]{2,}',
            fact
        )

        candidate_projects = {}

        for project_name in project_names:

            project_score = 0

            related_contents = [
                item["content"]
                for item in all_docs
                if item["metadata"].get("project_name", "") == project_name
            ]

            for content in related_contents:

                for part in chinese_parts:

                    if len(part) >= 2 and part in content:
                        project_score += len(part)

                # 额外检查 3 字关键词
                for i in range(len(part) - 2):
                    keyword = part[i:i + 3]

                    if keyword in content:
                        project_score += 2

            if project_score > 0:
                candidate_projects[project_name] = project_score

        if candidate_projects:

            matched_projects = [
                max(
                    candidate_projects,
                    key=candidate_projects.get
                )
            ]

    # =========================================================
    # 第五部分：建立候选证据
    # =========================================================

    candidates = {}

    # 5.1 加入向量检索结果
    for doc, score in vector_results:

        content = doc.page_content.strip()

        if not content:
            continue

        metadata = doc.metadata or {}

        project_name = (
            metadata.get("project_name")
            or infer_project_name(content)
            or "福建非遗"
        )

        key = content

        candidates[key] = {
            "content": content,
            "source": metadata.get("source", "未知来源"),
            "url": metadata.get("url", ""),
            "project_name": project_name,
            "score": float(score),
            "vector_score": float(score),
            "keyword_score": 0.0,
            "project_match": False
        }

    # =========================================================
    # 5.2 加入全文关键词检索结果
    # =========================================================

    chinese_parts = re.findall(
        r'[\u4e00-\u9fff]{2,}',
        fact
    )

    # 提取一些有价值的 2～4 字关键词
    keywords = set()

    for part in chinese_parts:

        if len(part) <= 4:
            keywords.add(part)

        else:
            # 生成 3 字关键词
            for i in range(len(part) - 2):
                keywords.add(part[i:i + 3])

    # 去掉过短、过泛的词
    stop_words = {
        "主要",
        "属于",
        "是一种",
        "具有",
        "文化",
        "传统",
        "中国",
        "福建",
        "非物",
        "质文",
        "列入",
        "代表"
    }

    keywords = {
        x for x in keywords
        if x not in stop_words
    }

    for item in all_docs:

        content = item["content"]
        metadata = item["metadata"]

        project_name = (
            metadata.get("project_name")
            or infer_project_name(content)
            or "福建非遗"
        )

        # -----------------------------------------------------
        # 计算关键词匹配程度
        # -----------------------------------------------------

        matched_keywords = []

        for keyword in keywords:

            if keyword in content:
                matched_keywords.append(keyword)

        keyword_score = 0.0

        if keywords:
            keyword_score = (
                len(matched_keywords) / len(keywords)
            )

        # -----------------------------------------------------
        # 判断是否属于当前项目
        # -----------------------------------------------------

        project_match = False

        if matched_projects:

            if project_name in matched_projects:
                project_match = True

        # 如果事实中直接出现项目名，也进行严格匹配
        for project_name2 in matched_projects:

            if project_name2 in content:
                project_match = True

        # -----------------------------------------------------
        # 只有关键词有一定匹配，
        # 或者属于当前项目，才加入候选
        # -----------------------------------------------------

        if (
            keyword_score > 0
            or project_match
        ):

            key = content

            if key not in candidates:

                candidates[key] = {
                    "content": content,
                    "source": metadata.get(
                        "source",
                        "未知来源"
                    ),
                    "url": metadata.get(
                        "url",
                        ""
                    ),
                    "project_name": project_name,
                    "score": 0.0,
                    "vector_score": 0.0,
                    "keyword_score": keyword_score,
                    "project_match": project_match
                }

            else:

                candidates[key]["keyword_score"] = max(
                    candidates[key]["keyword_score"],
                    keyword_score
                )

                if project_match:
                    candidates[key]["project_match"] = True

                if (
                    not candidates[key]["source"]
                    or candidates[key]["source"] == "未知来源"
                ):
                    candidates[key]["source"] = metadata.get(
                        "source",
                        "未知来源"
                    )

    # =========================================================
    # 第六部分：重新计算最终排序分数
    # =========================================================

    final_candidates = []

    for item in candidates.values():

        vector_score = item.get(
            "vector_score",
            0.0
        )

        keyword_score = item.get(
            "keyword_score",
            0.0
        )

        project_match = item.get(
            "project_match",
            False
        )

        # 基础分
        final_score = (
            vector_score * 0.55
            + keyword_score * 0.45
        )

        # 当前项目匹配给予明显加权
        if project_match:
            final_score += 0.35

        # 限制最高分
        final_score = min(
            final_score,
            1.0
        )

        item["score"] = final_score

        # =====================================================
        # 如果已经明确知道项目，
        # 非当前项目证据不进入最终结果
        # =====================================================

        if matched_projects:

            if item["project_name"] not in matched_projects:

                continue

        # =====================================================
        # 最低相关度判断
        #
        # 项目完全匹配时，即使向量分低，
        # 也允许作为证据进入。
        # =====================================================

        if (
            item["score"] >= threshold
            or item["project_match"]
        ):
            final_candidates.append(item)

    # =========================================================
    # 第七部分：排序
    # =========================================================

    final_candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    # 最多返回 5 条证据
    final_candidates = final_candidates[:5]

    # =========================================================
    # 第八部分：去重
    # =========================================================

    return deduplicate_evidence(final_candidates)


# =========================================================
# 单条事实核验
# =========================================================
def check_single_fact(fact, evidence):
    llm = load_llm()
    if not evidence:
        return {
            "status":"⚪ 无据",
            "fact":fact,
            "reason":"知识库未检索到相关权威证据",
            "suggestion":"补充权威资料再核验",
            "confidence":0.9,
            "evidence":[]
        }
    ev_text = ""
    for idx,ev in enumerate(evidence,1):
        ev_text += f"""【证据{idx}】
原文：{ev['content']}
来源：{ev['source']}
相关度：{ev['score']:.2f}
"""
    prompt = f"""
你是严谨福建非遗事实核验专家。
待核验事实：
{fact}

知识库证据：
{ev_text}

状态四选一：🟢 有据 / 🔴 疑误 / 🟡 存疑 / ⚪ 无据

判定规则：
1. 只有证据与待核验事实的关键内容一致，并且能够直接支持事实，才能判定为“🟢 有据”。
2. 如果证据与事实存在明确冲突，必须判定为“🔴 疑误”。
3. 特别注意比较年份、地点、项目名称、等级、人物、数量、材料等具体信息。
4. “同一个非遗项目”不代表“事实正确”。如果项目相同但关键事实不同，必须判定为“🔴 疑误”。
5. 如果证据相关但无法确认事实是否正确，判定为“🟡 存疑”。
6. 如果没有足够相关证据支持或反驳，判定为“⚪ 无据”。

例如：
事实：泉州拍胸舞于2008年列入首批国家级非物质文化遗产代表性名录。
证据：泉州拍胸舞于2006年列入首批国家级非物质文化遗产代表性名录。
结论必须是“🔴 疑误”，因为年份2008与2006存在明确冲突。
reason中必须指出这一具体冲突。
只输出JSON，不要多余文字。
{{
"status":"",
"reason":"一句话说明判断依据",
"suggestion":"如果事实存在错误，必须给出具体修改后的正确表述；如果事实正确，才写无需修改。",
"confidence":0~1小数
}}
"""
    try:
        resp = llm.invoke(prompt)
        cnt = re.sub(r"```json|```", "", resp.content.strip())
        jdata = json.loads(cnt)

        status = jdata.get("status", "🟡 存疑")
        reason = jdata.get("reason", "")
        suggestion = jdata.get("suggestion", "")

        # 如果判定为“疑误”，强制保证建议不能是“无需修改”
        if status == "🔴 疑误":
            suggestion = f"原事实存在错误，请根据权威证据进行修改。{reason}"

        return {
            "status": status,
            "fact": fact,
            "reason": reason,
            "suggestion": suggestion,
            "confidence": float(jdata.get("confidence", 0.5)),
            "evidence": evidence
        }

    except Exception as e:
        return {
            "status": "🟡 存疑",
            "fact": fact,
            "reason": f"调用模型异常:{str(e)}",
            "suggestion": "重新核验",
            "confidence": 0.4,
            "evidence": evidence
        }

# =========================================================
# 展示证据组件
# =========================================================
def show_sources(ev_list):
    if not ev_list:
        return
    with st.expander("📚查看权威证据"):
        for i,ev in enumerate(ev_list,1):
            st.markdown(f"**证据{i}**")
            st.markdown(f'<div class="evidence-box">{ev["content"]}</div>',unsafe_allow_html=True)
            st.caption(f"来源:{ev['source']}｜相关度:{ev['score']:.2f}｜项目:{ev['project_name']}")
            if ev.get("url"):
                st.markdown(f"🔗[{ev['url']}]({ev['url']})")


# =========================================================
# 渲染单条结果
# =========================================================
def show_result(result):
    st.write(f"**状态：{result['status']}**")
    st.markdown(f"> 📌事实：{result['fact']}")
    st.markdown(f"💡核验说明：{result['reason']}")
    st.markdown(f"✏️建议：{result['suggestion']}")
    st.progress(min(max(result["confidence"],0),1),text=f"置信度 {result['confidence']:.0%}")
    show_sources(result["evidence"])


# =========================================================
# 主输入区
# =========================================================
st.markdown("## ✍️ 输入待核验的 AIGC 非遗文本")
st.caption("输入福建非遗文本，系统自动拆分多条事实逐条核验")

demo_text = """惠安石雕是以福建福州市为核心区域的传统雕刻技艺，属于国家级非物质文化遗产。
其主要采用花岗岩进行雕刻，技法包括圆雕、浮雕、沉雕和影雕，广泛应用于寺庙、古民居、园林和碑石等。
福州脱胎漆器与景德镇瓷器、北京景泰蓝并称中国传统工艺三宝。
"""

if st.button("🎬加载演示案例",use_container_width=True):
    st.session_state["input_text"] = demo_text

input_text = st.text_area(
    "待核验文本",
    value=st.session_state.get("input_text",""),
    height=180,
    placeholder="粘贴你要校验的非遗文本……",
    key="main_input"
)

st.markdown("### ⚙️核验参数")
threshold = st.slider(
    "证据相关度阈值",
    min_value=0.50,
    max_value=0.90,
    value=0.65,
    step=0.05,
    help="向量检索最低相关度阈值"
)

check_btn = st.button("🔍开始勘误校验",type="primary",use_container_width=True)

if check_btn:
    if not input_text.strip():
        st.warning("请输入待核验文本")
        st.stop()
    with st.spinner("🤖正在拆分事实……"):
        facts = split_fact_sentences(input_text)
    if not facts:
        st.warning("识别不到可核验事实，请修改文本")
        st.stop()
    st.success(f"✅识别 {len(facts)} 条待核验事实")

    res_list = []
    bar = st.progress(0,"正在核验……")
    for idx,fact in enumerate(facts):
        evd = get_evidence(fact,threshold)
        one = check_single_fact(fact,evd)
        res_list.append(one)
        bar.progress((idx+1)/len(facts),text=f"核验 {idx+1}/{len(facts)}")
    bar.empty()

    # =========================
    # 核验统计
    # =========================

    # =========================
    # 核验统计
    # =========================
    stat = {
        "有据": 0,
        "疑误": 0,
        "存疑": 0,
        "无据": 0
    }

    for r in res_list:
        raw_status = str(r.get("status", "")).strip()

        # 同时兼容：
        # 🟢
        # 🟢 有据
        # 有据
        # 🔴
        # 🔴 疑误
        # 疑误
        if "🟢" in raw_status or "有据" in raw_status:
            s = "有据"

        elif "🔴" in raw_status or "疑误" in raw_status:
            s = "疑误"

        elif "🟡" in raw_status or "存疑" in raw_status:
            s = "存疑"

        elif "⚪" in raw_status or "无据" in raw_status:
            s = "无据"

        else:
            s = "存疑"

        stat[s] += 1

    st.divider()

    st.markdown("## 📊 核验统计")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("### 🟢 有据")
        st.markdown(
            f'<div class="stat-number">{stat["有据"]}</div>',
            unsafe_allow_html=True
        )

    with c2:
        st.markdown("### 🔴 疑误")
        st.markdown(
            f'<div class="stat-number">{stat["疑误"]}</div>',
            unsafe_allow_html=True
        )

    with c3:
        st.markdown("### 🟡 存疑")
        st.markdown(
            f'<div class="stat-number">{stat["存疑"]}</div>',
            unsafe_allow_html=True
        )

    with c4:
        st.markdown("### ⚪ 无据")
        st.markdown(
            f'<div class="stat-number">{stat["无据"]}</div>',
            unsafe_allow_html=True
        )
    st.divider()
    st.markdown(f"## 📋核验详情（共{len(res_list)}条）")
    #排序：疑误优先展示，比赛效果好
    order_weight = {" 疑误":0," 存疑":1," 无据":2," 有据":3}
    sorted_res = sorted(res_list,key=lambda x:order_weight.get(x["status"],99))
    for num,item in enumerate(sorted_res,1):
        with st.container(border=True):
            st.markdown(f"### 第{num}条")
            show_result(item)
        st.write("")

st.divider()
st.caption("🏮闽遗智鉴｜福建非遗AIGC内容事实核验与权威证据溯源系统")
st.caption("核验仅供参考，请以官方权威资料为准。")
