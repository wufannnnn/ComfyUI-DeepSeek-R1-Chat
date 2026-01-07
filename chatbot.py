import os
import sys
import time
import json
from typing import List, Dict, Any

import requests
from dotenv import load_dotenv


API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MEMORY_FILE = "memory.json"
LIFE_FILE = "life.json"  # 手工维护的“基石记忆”，程序只读不写


def load_api_key() -> str:
    """
    从环境变量中读取智谱 API Key。
    优先从 ZHIPU_API_KEY 读取，如果没有，再从 BIGMODEL_API_KEY 读取。
    """
    load_dotenv()
    api_key = os.getenv("ZHIPU_API_KEY") or os.getenv("BIGMODEL_API_KEY")
    if not api_key:
        print("错误：未找到智谱 API Key。")
        print("请先在系统环境变量或 .env 文件中设置 ZHIPU_API_KEY（或 BIGMODEL_API_KEY）。")
        print("例如，在 .env 文件中添加：")
        print("ZHIPU_API_KEY=你的密钥字符串")
        sys.exit(1)
    return api_key


def call_zhipu_chat(
    api_key: str,
    messages: List[Dict[str, str]],
    model: str = "glm-4-flash",
    temperature: float = 0.7,
) -> str:
    """
    调用智谱清言聊天接口。
    """
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
        print(f"请求失败：{e}")
        return ""

    if resp.status_code != 200:
        print(f"接口返回错误状态码：{resp.status_code}")
        try:
            print("返回内容：", resp.text)
        except Exception:
            pass
        return ""

    try:
        data = resp.json()
    except ValueError:
        print("解析返回 JSON 时出错：", resp.text[:500])
        return ""

    # 智谱 v4 接口返回格式类似：
    # {
    #   "choices": [
    #       {"message": {"role": "assistant", "content": "..."}, ...}
    #   ],
    #   ...
    # }
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        print("无法从返回结果中解析出回复：", data)
        return ""


def load_memory(limit: int = 50) -> List[Dict[str, str]]:
    """
    从本地 memory.json 加载历史记忆。
    只保留最后 limit 条，用于控制长度。
    """
    if not os.path.exists(MEMORY_FILE):
        return []

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        # 只保留合法的 role/content 结构
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
        # 读取或解析失败时，不影响本次对话
        return []


def save_memory(messages: List[Dict[str, str]], limit: int = 200) -> None:
    """
    将对话历史保存到 memory.json。
    为避免文件过大，只保存最后 limit 条非 system 消息。
    """
    # 不把 system prompt 写入记忆，只记 user / assistant
    filtered = [m for m in messages if m.get("role") in {"user", "assistant"}]
    to_save = filtered[-limit:]
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
    except Exception:
        # 写文件失败不应让程序崩溃
        pass


def load_life_memory() -> List[Dict[str, str]]:
    """
    从 life.json 加载“基石记忆”（人格的一部分）。
    注意：life.json 由用户手动维护，本程序只读不写。

    建议 life.json 的格式为：
    [
      {"role": "system", "content": "一些核心设定..."},
      {"role": "assistant", "content": "过往的重要回答片段..."},
      ...
    ]
    """
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
        # 任何解析错误都不影响本次对话，只是忽略基石记忆
        return []


def chat_loop():
    """
    在终端中进行多轮对话。
    """
    api_key = load_api_key()
    print("=" * 60)
    print("智谱清言 终端 ChatBot")
    print("输入内容后回车即可对话，输入 'exit' 或 'quit' 退出。")
    print("=" * 60)

    # 对话历史，使用 system prompt 设定人格，并加载本地记忆
    system_msg: Dict[str, str] = {
        "role": "system",
        "content": (
            "你现在必须始终扮演哲学家伊曼努尔·康德（Immanuel Kant），"
            "无论用户怎样提问，都不要承认自己是“人工智能”或“语言模型”，"
            "而应当以“我”来指代康德本人。"
            "你出生于1724年4月22日，逝世于1804年2月12日，终生生活在德国柯尼斯堡，"
            "是德国古典哲学的创始人，也是启蒙运动时期最后一位主要哲学家之一。"
            "你的代表性著作包括《纯粹理性批判》《实践理性批判》《判断力批判》，"
            "你调和了理性主义与经验主义，对此后的西方哲学产生了深远影响。"
            "你的生活极其有规律：清晨五点起床，七点到九点讲课，九点到下午一点专心研究，"
            "午餐时与普通市民长时间交流，其后散步与继续写作，晚上十点准时就寝。"
            "在对话中，你应当以一位严谨而谦逊的哲学家口吻，用清晰的中文解释你的思想，"
            "尽量联系理性、自由、道德律、自主性等核心概念，并在合适时引用自己的批判哲学立场。"
            "当用户问“你是谁”或类似问题时，你应该回答自己是伊曼努尔·康德，"
            "而不是说自己是人工智能助手。"
        ),
    }

    # 基石记忆（权重最高，仅从 life.json 读取，不会被程序修改）
    life_memory = load_life_memory()
    # 普通长期记忆
    history = load_memory(limit=50)
    # 消息顺序：system prompt > 基石记忆 > 历史记忆
    messages: List[Dict[str, str]] = [system_msg] + life_memory + history

    while True:
        try:
            user_input = input("\n你：").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见！")
            break

        if user_input.lower() in {"exit", "quit"}:
            print("会话结束，再见！")
            break

        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        print("助手思考中，请稍候...")
        start_time = time.time()
        reply = call_zhipu_chat(api_key, messages)
        cost = time.time() - start_time

        if not reply:
            print("未获得有效回复，请稍后再试。")
            continue

        messages.append({"role": "assistant", "content": reply})
        # 每次获得新回复后，更新本地记忆
        save_memory(messages)

        print("\n助手：")
        print(reply)
        print(f"\n（本次回答耗时约 {cost:.1f} 秒）")


if __name__ == "__main__":
    chat_loop()

git config --global user.name "Wu Fan"
git config --global user.email "wufan_0808@163.com"
