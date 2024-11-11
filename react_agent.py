system_prompt = """
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


from loguru import logger
from vllm_llm import LLM

client = LLM()
class Agent:
    def __init__(self, client: LLM, system: str = "") -> None:
        self.client = client
        self.system = system
        self.messages: list = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message=""):
        if message:
            self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        return client.agent_answer(self.messages)


import re
import httpx
def wikipedia(q):
    url = f"https://baike.baidu.com/api/openapi/BaikeLemmaCardApi?scope=103&format=json&appid=379020&bk_key={q}&bk_length=600"
    response = httpx.get(url)
    data = response.json()
    if 'abstract' in data:
        return data['abstract']
    else:
        return "未找到相关信息"

def calculate(operation: str) -> float:
    return eval(operation) # eval 表示 字符串 转换为 表达式
#
def loop(max_iterations=10, query: str = ""):

    agent = Agent(client=client, system=system_prompt)

    tools = ["calculate", "wikipedia"]

    next_prompt = query

    i = 0
  
    while i < max_iterations:
        i += 1
        result = agent(next_prompt)
        logger.info(f"result: {result}")

        if "PAUSE" in result and "Action" in result:
            action = re.findall(r"Action: ([a-z_]+): (.+)", result, re.IGNORECASE)
            logger.info(f"action: {action}")
            chosen_tool = action[0][0]
            arg = action[0][1]

            if chosen_tool in tools:
                result_tool = eval(f"{chosen_tool}('{arg}')")
                next_prompt = f"Observation: {result_tool}"

            else:
                next_prompt = "Observation: Tool not found"

            logger.info(f"next_prompt: {next_prompt}")
            continue

        if "Answer" in result:
            break

if __name__ == "__main__":
    logger.info(loop(query="地球距离太阳的距离乘以2等于多少?"))