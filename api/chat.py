import os
import uuid
import requests
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List
from .database import get_db
from .db import Conversation, FAQ   # 导入 FAQ 模型

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    user_id: str
    session_id: str

# 读取知识库系统提示
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
knowledge_path = os.path.join(base_dir, "knowledge_base.txt")
with open(knowledge_path, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read().strip()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

def get_conversation_history(db: Session, user_id: str, session_id: str, limit: int = 5) -> List[dict]:
    history = db.query(Conversation).filter(
        Conversation.user_id == user_id,
        Conversation.session_id == session_id
    ).order_by(Conversation.timestamp.asc()).limit(limit).all()
    messages = []
    for h in history:
        messages.append({"role": "user", "content": h.user_message})
        messages.append({"role": "assistant", "content": h.ai_reply})
    return messages

def find_faq_match(db: Session, user_message: str) -> str | None:
    """在 faqs 表中查找匹配的问题，返回答案；未找到返回 None"""
    msg_lower = user_message.lower().strip()
    # 获取所有 FAQ
    all_faqs = db.query(FAQ).all()
    for faq in all_faqs:
        if faq.question.lower() in msg_lower or msg_lower in faq.question.lower():
            return faq.answer
    return None

@router.post("/chat")
def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    # 1. 优先匹配 FAQ 表
    faq_answer = find_faq_match(db, request.message)
    if faq_answer:
        ai_message = faq_answer
        # 仍然保存对话记录
        conv = Conversation(
            id=str(uuid.uuid4()),
            user_id=request.user_id,
            session_id=request.session_id,
            user_message=request.message,
            ai_reply=ai_message
        )
        db.add(conv)
        db.commit()
        return {"response": ai_message}

    # 2. 没有匹配到 FAQ 时，走原有流程（历史 + DeepSeek API）
    history_messages = get_conversation_history(db, request.user_id, request.session_id, limit=5)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": request.message})

    if not DEEPSEEK_API_KEY:
        ai_message = fallback_reply(request.message)
    else:
        try:
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "deepseek-chat",
                "messages": messages,
                "stream": False
            }
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            ai_message = data['choices'][0]['message']['content']
        except Exception as e:
            print(f"DeepSeek API 调用失败: {e}")
            ai_message = fallback_reply(request.message)

    # 保存对话记录
    conv = Conversation(
        id=str(uuid.uuid4()),
        user_id=request.user_id,
        session_id=request.session_id,
        user_message=request.message,
        ai_reply=ai_message
    )
    db.add(conv)
    db.commit()

    return {"response": ai_message}

def fallback_reply(message: str) -> str:
    msg_lower = message.lower()
    if "无法开机" in msg_lower or "开不了机" in msg_lower:
        return "请检查电池是否安装正确，电池是否有电，尝试长按电源键3秒。"
    elif "图传" in msg_lower and ("弱" in msg_lower or "信号" in msg_lower):
        return "请检查天线是否展开，周围是否有强电磁干扰，尝试切换频段。"
    elif "遥控器" in msg_lower and "配对" in msg_lower:
        return "请在无人机开机状态下，长按遥控器配对键5秒，直到听到提示音。"
    elif "保修" in msg_lower:
        return "整机保修1年，电池保修6个月，人为损坏不在保修范围内。"
    else:
        return f"您好，我是X-Drone售后助手。您的问题是：“{message}”。我会尽力为您解答。如需更多帮助，请拨打客服热线。"