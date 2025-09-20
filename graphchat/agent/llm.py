from langchain_community.chat_models import ChatZhipuAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv


load_dotenv()


model_gemini = "gemini-2.0-flash-exp"
chat_model = ChatGoogleGenerativeAI(model = model_gemini,api_key = os.getenv("GEMINI_API_KEY"))


model = "glm-4-flash"
Zhipullm = ChatZhipuAI(model=model, api_key=os.getenv("CHATGLM_API_KEY"))



model_deep="deepseek/deepseek-r1:free"
deep_chat = ChatOpenAI(model=model_deep,api_key=os.getenv("DEEPSEEK_API_KEY"),base_url="https://openrouter.ai/api/v1")

