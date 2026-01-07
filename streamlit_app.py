"""
Streamlit Web 版本的 ChatBot
复用 chatbot.py 中的核心逻辑，提供 Web 界面
"""
import os
import json
import time
from typing import List, Dict, Any

import streamlit as st
import requests

# 复用 chatbot.py 中的常量和函数逻辑
API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MEMORY_FILE = "memory.json"
LIFE_FILE = "life.json"


def load_api_key() -> str:
    """
    从 Streamlit secrets 或环境变量中读取智谱 API Key。
    在 Streamlit Cloud 上使用 st.secrets，本地开发时使用环境变量。
    """
    # 优先从 Streamlit secrets 读取（用于 Streamlit Cloud）
    api_key = None
    try:
        api_key = st.secrets.get("ZHIPU_API_KEY") or st.secrets.get("BIGMODEL_API_KEY")
    except Exception:
        pass
    
    # 如果 secrets 中没有，尝试从环境变量读取（用于本地开发）
    if not api_key:
        api_key = os.getenv("ZHIPU_API_KEY") or os.getenv("BIGMODEL_API_KEY")
    
    if not api_key:
        st.error("❌ 未找到智谱 API Key。")
        st.info(
            "**在 Streamlit Cloud 上：**\n"
            "请在 App Settings → Secrets 中添加：\n"
            "```toml\n"
            "ZHIPU_API_KEY = '你的密钥'\n"
            "```\n\n"
            "**本地开发时：**\n"
            "请在 `.env` 文件中添加：\n"
            "```\n"
            "ZHIPU_API_KEY=你的密钥\n"
            "```"
        )
        st.stop()
    
    return api_key


def call_zhipu_chat(
    api_key: str,
    messages: List[Dict[str, str]],
    model: str = "glm-4-flash",
    temperature: float = 0.7,
) -> str:
    """调用智谱清言聊天接口。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    
    try:
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=60)
    except requests.RequestException as e:
        st.error(f"请求失败：{e}")
        return ""
    
    if resp.status_code != 200:
        st.error(f"接口返回错误状态码：{resp.status_code}")
        try:
            st.error(f"返回内容：{resp.text[:500]}")
        except Exception:
            pass
        return ""
    
    try:
        data = resp.json()
    except ValueError:
        st.error(f"解析返回 JSON 时出错：{resp.text[:500]}")
        return ""
    
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        st.error(f"无法从返回结果中解析出回复：{data}")
        return ""


def load_memory(limit: int = 50) -> List[Dict[str, str]]:
    """从本地 memory.json 加载历史记忆。"""
    if not os.path.exists(MEMORY_FILE):
        return []
    
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        messages: List[Dict[str, str]] = []
        for item in data[-limit:]:
            if (
                isinstance(item, dict)
                and isinstance(item.get("role"), str)
                and isinstance(item.get("content"), str)
            ):
                messages.append({"role": item["role"], "content": item["content"]})
        return messages
    except Exception:
        return []


def save_memory(messages: List[Dict[str, str]], limit: int = 200) -> None:
    """将对话历史保存到 memory.json。"""
    filtered = [m for m in messages if m.get("role") in {"user", "assistant"}]
    to_save = filtered[-limit:]
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def load_life_memory() -> List[Dict[str, str]]:
    """从 life.json 加载“基石记忆”（人格的一部分）。"""
    if not os.path.exists(LIFE_FILE):
        return []
    
    try:
        with open(LIFE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        messages: List[Dict[str, str]] = []
        for item in data:
            if (
                isinstance(item, dict)
                and isinstance(item.get("role"), str)
                and isinstance(item.get("content"), str)
            ):
                messages.append({"role": item["role"], "content": item["content"]})
        return messages
    except Exception:
        return []


# Streamlit 页面配置
st.set_page_config(
    page_title="DeepSeek Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 标题和说明
st.title("💬 DeepSeek Chat")
st.caption("基于智谱清言的对话机器人，支持基石记忆和长期记忆")

# 加载 API Key
api_key = load_api_key()

# System prompt（与 chatbot.py 保持一致）
system_msg: Dict[str, str] = {
    "role": "system",
    "content": (
        "你是一个用简体中文交流的助手，回答应清晰、简洁、有条理，"
        "避免无意义的客套，遇到不清楚的需求先询问澄清。"
        "人格、口吻、具体示例请严格遵循后续的 life.json 基石记忆和对话历史。"
        "当用户问'你是谁'或类似问题时，你必须优先参考 life.json 中的示例回答，"
        "不得回答'我是一个AI助手'或'我是一个语言模型'等表述，而是使用基石记忆中体现出来的身份与自我描述。"
    ),
}

# 初始化会话状态
if "messages" not in st.session_state:
    # 加载基石记忆和历史记忆
    life_memory = load_life_memory()
    history = load_memory(limit=50)
    # 消息顺序：system prompt > 基石记忆 > 历史记忆
    st.session_state.messages = [system_msg] + life_memory + history
    st.session_state.api_key = api_key

# 显示历史对话
for msg in st.session_state.messages:
    # 跳过 system 消息，不显示在界面上
    if msg["role"] == "system":
        continue
    
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("输入你的问题..."):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 调用 API 获取回复
    with st.chat_message("assistant"):
        with st.spinner("助手思考中，请稍候..."):
            start_time = time.time()
            reply = call_zhipu_chat(api_key, st.session_state.messages)
            cost = time.time() - start_time
        
        if reply:
            st.markdown(reply)
            st.caption(f"⏱️ 本次回答耗时约 {cost:.1f} 秒")
            # 添加助手回复到消息列表
            st.session_state.messages.append({"role": "assistant", "content": reply})
            # 保存记忆
            save_memory(st.session_state.messages)
        else:
            st.error("未获得有效回复，请稍后再试。")

# 侧边栏：清空对话按钮
with st.sidebar:
    st.header("⚙️ 设置")
    if st.button("🗑️ 清空对话历史", type="secondary"):
        # 只清空 user/assistant 消息，保留 system 和基石记忆
        life_memory = load_life_memory()
        st.session_state.messages = [system_msg] + life_memory
        # 清空 memory.json
        try:
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                    json.dump([], f)
        except Exception:
            pass
        st.rerun()
    
    st.divider()
    st.info(
        "💡 **提示：**\n\n"
        "- 基石记忆（life.json）不会被清空\n"
        "- 对话历史会自动保存到 memory.json\n"
        "- 刷新页面不会丢失对话历史"
    )
