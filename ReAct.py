from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from loguru import logger
import httpx
import re
from abc import ABC, abstractmethod
from urllib.parse import quote
from bs4 import BeautifulSoup

# System Prompt
SYSTEM_PROMPT = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

wikipedia:
e.g. wikipedia: Django
Returns a summary from searching Wikipedia

Always look things up on Wikipedia if you have the opportunity to do so.

Example session:

Question: What is the capital of France?
Thought: I should look up France on Wikipedia
Action: wikipedia: France
PAUSE 

You will be called again with this:

Observation: France is a country. The capital is Paris.
Thought: I think I have found the answer
Action: Paris.
You should then call the appropriate action and determine the answer from the result

You then output:

Answer: The capital of France is Paris

Example session

Question: What is the mass of Earth times 2?
Thought: I need to find the mass of Earth on Wikipedia
Action: wikipedia : mass of earth
PAUSE

You will be called again with this: 

Observation: mass of earth is 1,1944×10e25

Thought: I need to multiply this by 2
Action: calculate: 5.972e24 * 2
PAUSE

You will be called again with this: 

Observation: 1,1944×10e25

If you have the answer, output it as the Answer.

Answer: The mass of Earth times 2 is 1,1944×10e25.

Now it's your turn:
""".strip()


@dataclass
class Config:
    MAX_ITERATIONS: int = 10
    SEARCH_API: str = "https://cn.bing.com/search"
    SEARCH_PARAMS: Dict[str, Any] = field(default_factory=lambda: {
        "form": "QBRE",
        "cc": "CN",
        "rdr": "1",
        "lq": "0"
    })

class Tool(ABC):
    @abstractmethod
    def __call__(self, input_text: str) -> str:
        pass

class BingSearch(Tool):
    def __init__(self, config: Config):
        self.config = config
        self.client = httpx.Client(
            timeout=10.0,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8"
            }
        )
    
    def __call__(self, query: str) -> str:
        try:
            params = {**self.config.SEARCH_PARAMS, "q": query}
            response = self.client.get(self.config.SEARCH_API, params=params)
            response.raise_for_status()
            
            # 使用BeautifulSoup解析搜索结果
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 提取搜索结果摘要
            results = []
            for item in soup.select('.b_algo'):
                title = item.select_one('h2')
                desc = item.select_one('.b_caption p')
                if title and desc:
                    results.append(f"{title.get_text().strip()}: {desc.get_text().strip()}")
                if len(results) >= 5:  # 只返回前5个结果
                    break
                    
            return "\n".join(results) if results else "未找到相关结果"
            
        except Exception as e:
            logger.error(f"Search API error: {e}")
            return f"搜索失败: {str(e)}"

class Calculator(Tool):
    def __call__(self, expression: str) -> str:
        try:
            # 使用更安全的eval
            result = eval(expression, {"__builtins__": {}}, {})
            return str(result)
        except Exception as e:
            logger.error(f"Calculate error: {e}")
            return f"计算失败: {str(e)}"

class Agent:
    def __init__(self, llm_client: Any, system: str = "") -> None:
        self.llm_client = llm_client
        self.messages: List[Dict[str, str]] = []
        if system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message: str = "") -> str:
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self) -> str:
        return self.llm_client.agent_answer(self.messages)

class ReActLoop:
    def __init__(self, config: Config):
        self.config = config
        self.tools: Dict[str, Tool] = {
            "wikipedia": BingSearch(config),  # 使用必应搜索替代
            "calculate": Calculator()
        }
    
    def extract_action(self, text: str) -> Optional[tuple[str, str]]:
        if match := re.findall(r"Action: ([a-z_]+): (.+)", text, re.IGNORECASE):
            return match[0]
        return None

    def run(self, agent: Agent, query: str) -> str:
        next_prompt = query
        
        for i in range(self.config.MAX_ITERATIONS):
            result = agent(next_prompt)
            logger.info(f"Iteration {i+1} result: {result}")

            if "Answer" in result:
                return result

            if "PAUSE" not in result or "Action" not in result:
                logger.warning("Invalid response format")
                continue

            action = self.extract_action(result)
            if not action:
                logger.warning("Failed to extract action")
                continue

            tool_name, arg = action
            if tool := self.tools.get(tool_name):
                result_tool = tool(arg)
                next_prompt = f"Observation: {result_tool}"
            else:
                next_prompt = f"Observation: Tool '{tool_name}' not found"

            logger.info(f"Next prompt: {next_prompt}")

        return "达到最大迭代次数限制"

def main():
    config = Config()
    llm_client = LLM()
    agent = Agent(llm_client, system=SYSTEM_PROMPT)
    loop = ReActLoop(config)
    
    query = "复旦大学计算机学院张奇教授今年几岁"
    result = loop.run(agent, query)
    logger.info(f"Final result: {result}")

if __name__ == "__main__":
    from vllm_llm import LLM
    main()