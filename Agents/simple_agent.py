import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# ─────────────────────────────────────────────
# 1. ENVIRONMENT SETUP
# ─────────────────────────────────────────────
# Load variables from the .env file into the process environment.
# This keeps secrets (API keys) out of source code.
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")  # Retrieve the OpenAI API key from the environment

llm_name = "gpt-3.5-turbo"  # The LLM model we'll send requests to

# Initialise the OpenAI client with our API key
client = OpenAI(api_key=openai_key)


# ─────────────────────────────────────────────
# 2. AGENT CLASS
# ─────────────────────────────────────────────
class Agent:
    """
    A simple conversational agent that maintains a running message history.

    Each call to the agent:
      1. Appends the user message to the history.
      2. Sends the full history to the LLM (so the model has full context).
      3. Appends the assistant reply to the history.
      4. Returns the reply.

    This 'stateful' design is what allows the agent to reason across
    multiple turns (Thought → Action → Observation → Answer loop).
    """

    def __init__(self, system=""):
        """
        Args:
            system (str): Optional system prompt that shapes the agent's
                          behaviour and knowledge of available tools.
        """
        self.system = system
        self.messages = []  # Stores the full conversation history

        # If a system prompt is provided, add it as the first message
        if system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        """
        Send a user message to the agent and get a reply.

        Args:
            message (str): The user's input (question or observation).

        Returns:
            str: The assistant's response.
        """
        # Add the incoming user message to the conversation history
        self.messages.append({"role": "user", "content": message})

        # Call the LLM with the full history and get a response
        result = self.execute()

        # Store the assistant's reply so future turns have full context
        self.messages.append({"role": "assistant", "content": result})

        return result

    def execute(self):
        """
        Send the current message history to the OpenAI API and return
        the model's text response.
        """
        response = client.chat.completions.create(
            model=llm_name,
            temperature=0.0,   # 0 = deterministic / reproducible output
            messages=self.messages,
        )
        return response.choices[0].message.content


# ─────────────────────────────────────────────
# 3. SYSTEM PROMPT  (ReAct-style reasoning loop)
# ─────────────────────────────────────────────
# This prompt implements the ReAct pattern:
#   Thought  → the model reasons about what to do next
#   Action   → the model calls a tool, then pauses
#   PAUSE    → signals we should run the tool and feed back the result
#   Observation → the tool result we inject back into the conversation
#   Answer   → the model's final response to the user
prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

calculate:
e.g. calculate: 4 * 7 / 3
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

planet_mass:
e.g. planet_mass: Earth
returns the mass of a planet in the solar system

Example session:

Question: What is the combined mass of Earth and Mars?
Thought: I should find the mass of each planet using planet_mass.
Action: planet_mass: Earth
PAUSE

You will be called again with this:

Observation: Earth has a mass of 5.972 × 10^24 kg

You then output:

Answer: Earth has a mass of 5.972 × 10^24 kg

Next, call the agent again with:

Action: planet_mass: Mars
PAUSE

Observation: Mars has a mass of 0.64171 × 10^24 kg

You then output:

Answer: Mars has a mass of 0.64171 × 10^24 kg

Finally, calculate the combined mass.

Action: calculate: 5.972 + 0.64171
PAUSE

Observation: The combined mass is 6.61371 × 10^24 kg

Answer: The combined mass of Earth and Mars is 6.61371 × 10^24 kg
""".strip()


# ─────────────────────────────────────────────
# 4. TOOL FUNCTIONS  (the agent's "actions")
# ─────────────────────────────────────────────

def calculate(expression):
    """
    Safely evaluate a mathematical expression and return the result.

    Args:
        expression (str): A Python-compatible maths expression, e.g. "4 * 7 / 3".

    Returns:
        float | int: The computed result.

    Note:
        eval() is used here for simplicity in a learning context.
        In production, prefer a safe math parser (e.g. simpleeval).
    """
    return eval(expression)


def planet_mass(name):
    """
    Return the mass of a named planet in our solar system.

    Args:
        name (str): Planet name with correct capitalisation, e.g. "Earth".

    Returns:
        str: Human-readable mass string, e.g. "Earth has a mass of 5.972 × 10^24 kg".
    """
    # Masses in units of 10^24 kg (source: NASA planetary fact sheets)
    masses = {
        "Mercury": 0.33011,
        "Venus":   4.8675,
        "Earth":   5.972,
        "Mars":    0.64171,
        "Jupiter": 1898.19,
        "Saturn":  568.34,
        "Uranus":  86.813,
        "Neptune": 102.413,
    }
    return f"{name} has a mass of {masses[name]} × 10^24 kg"


# Map action names (as the LLM will write them) to the actual Python functions.
# The query loop uses this dict to dispatch the correct function at runtime.
known_actions = {
    "calculate": calculate,
    "planet_mass": planet_mass,
}


# ─────────────────────────────────────────────
# 5. ACTION PARSER
# ─────────────────────────────────────────────
# Regex that matches a line like:  Action: planet_mass: Earth
#   Group 1 → action name  (e.g. "planet_mass")
#   Group 2 → action input (e.g. "Earth")
action_re = re.compile(r"^Action: (\w+): (.*)$")


# ─────────────────────────────────────────────
# 6. INTERACTIVE QUERY LOOP
# ─────────────────────────────────────────────
def query_interactive():
    """
    Run an interactive ReAct agent loop in the terminal.

    How it works:
      • A fresh Agent is created with the ReAct system prompt.
      • The user types a question.
      • The agent replies with Thought / Action / PAUSE text.
      • If an Action is detected, the corresponding tool is called
        automatically and the result is fed back as an Observation.
      • The loop continues until no more actions are needed or
        max_turns is reached.
    """
    # Create a fresh agent loaded with the ReAct system prompt
    bot = Agent(prompt)

    # Ask the user how many conversation turns to allow before stopping
    max_turns = int(input("Enter the maximum number of turns: "))
    i = 0

    while i < max_turns:
        i += 1

        # Get the user's question for this turn
        question = input("You: ")

        # Send the question to the agent and print its initial response
        result = bot(question)
        print("Bot:", result)

        # ── Action detection ──────────────────────────────────────────
        # Scan each line of the response for an "Action: ..." pattern.
        actions = [
            action_re.match(line)
            for line in result.split("\n")
            if action_re.match(line)
        ]

        if actions:
            # Extract the action name and its input from the first match
            action, action_input = actions[0].groups()

            # Guard against the model hallucinating a tool we don't have
            if action not in known_actions:
                print(f"Unknown action: {action}: {action_input}")
                continue

            # Run the tool and display what we're doing
            print(f" -- running {action}({action_input})")
            observation = known_actions[action](action_input)
            print("Observation:", observation)

            # Feed the tool result back into the agent as an Observation
            # so it can continue reasoning toward a final Answer.
            next_prompt = f"Observation: {observation}"
            result = bot(next_prompt)
            print("Bot:", result)

        else:
            # No action found — the agent has produced its final Answer
            print("No actions to run.")
            break


# ─────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    query_interactive()