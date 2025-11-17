## About the Challenge
### Frontier - Fund Search

The Challenge

Build the underlying tools that give an AI agent the ability to discover Brazilian
investment funds based on user queries and return their CNPJs.
You should also create a thin agent layer that uses the tools you build. Keep in
mind that our focus is on the tools themselves, not the agent: the agent simply
serves as a natural language interface to your tools.
You decide what types of search criteria to support and how to handle them.
For inspiration, users might ask things like "what is the Bradesco gold fund?" or
"show me some funds that invest in Latin American tech", but these are just
examples: implement the fund-searching capabilities you find most interesting
or valuable.

### Requiremens

Your solution should include the following:
Data Collection: Download fund data from CVM (Portal Dados Abertos
CVM) and other sources you might find interesting (details below). Decide
what data is relevant for your implementation.
Tools and Agent: Build the tools you think are useful and expose them to a
simple agent so it can help find Brazilian funds based on user queries and
retrieve their CNPJs. The agent can be a CLI chatbot, have a UI, or
whatever interface you prefer. Our focus is on the fund finding capability;
the agent doesn't need to be particularly good at anything else.
Evaluation: Write comprehensive evals to measure the quality of your
answers and edge case handling. We care as much about how you evaluate
the tool's quality as we do about the tool itself.
Once you have completed this task, you should send us a GitHub repository
with your code and then present it in an online meeting with the team.
