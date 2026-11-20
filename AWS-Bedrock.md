1. What is Amazon Bedrock?

Bedrock is AWS’s managed service for using generative AI. You don’t host or train any AI models yourself. You call them through an AWS API, the same way you call S3 or DynamoDB.

It gives you:

Models from many providers (Anthropic Claude, Amazon Nova, Meta Llama, Mistral and others) behind one API
Knowledge Bases, so the AI can answer from your own documents
Agents, so the AI can take actions and call your systems
Guardrails, which filter harmful content and protect sensitive data

Security, IAM, scaling and logging are handled the AWS way.

2. Model (Foundation Model)

The model is the “brain”: a large language model that takes text in and gives text out.

Example: You send “Summarize this incident in 3 lines” plus the incident text, and Claude returns the summary.

By itself a model:

Knows only what it was trained on, so it has never seen your internal runbooks
Cannot take actions
Has no memory between calls

Knowledge Bases, Agents and Memory exist to fix exactly these gaps.

Model ID: each model has an identifier, like anthropic.claude-.... You pass it in your API call to say which model to use. In some regions you use an inference profile ID instead, which routes requests across regions for capacity.

3. Context (and the context window)

Context is everything the model can see in one request: your instructions, the conversation so far, any documents you attach, and the question.

The context window is the maximum size of that context, measured in tokens (roughly ¾ of a word each). Anything outside the window is invisible to the model.

Example:

System: You are an SRE assistant.
User: What is MTTR?
Assistant: Mean Time To Resolve...
User: And how do we improve it?   ← "it" only makes sense because the earlier messages are in the context

Key point: models are stateless. They remember nothing between calls. Every time, you (or Bedrock) must send the relevant history again. Sessions and memory are about managing this.

4. Knowledge Base, Vector DB and RAG
The problem

Your runbooks, RCAs and policies are not in the model’s training data. You could paste them into the prompt, but you have thousands of pages and the context window is limited.

RAG (Retrieval-Augmented Generation)

Retrieve only the relevant pieces first, then give them to the model along with the question.

A Knowledge Base is a managed RAG pipeline

Setup (once):

Data source: point it at an S3 bucket holding your PDFs, Word files and wiki exports (also Confluence, SharePoint and others).
Chunking: documents are split into small pieces (e.g. 300 tokens each).
Embedding: an embedding model (e.g. Titan Embeddings) turns each chunk into a list of numbers (a vector) that captures its meaning.
Vector DB: those vectors are stored in a vector database (OpenSearch Serverless, Aurora PostgreSQL with pgvector, Pinecone and others).

At question time:

The question “How do I escalate a P1?” is also converted into a vector.
The vector DB finds the chunks whose vectors are closest in meaning. This is semantic search, so it works even if your document says “Severity-1 escalation procedure” and never uses the word “P1”.
The top chunks are inserted into the prompt.
The model answers using them and can cite the source.
What is a vector DB?

A database built to store embeddings and quickly answer “which stored items are most similar to this one?” A normal database matches exact values. A vector DB matches meaning.

Since you already run Aurora PostgreSQL for your portal, note that Aurora with the pgvector extension is a supported vector store option for Knowledge Bases.

Example: with your runbooks in a Knowledge Base, “What do I do if the pod keeps crash-looping?” retrieves your own Kubernetes troubleshooting runbook and answers from it, instead of giving generic internet advice.

AWS has also recently added a Managed Knowledge Base with an agentic retriever. It can plan queries across knowledge bases, connect related concepts across documents, evaluate intermediate results and re-rank them before answering. 
awsinsider

5. Agent

A model answers. An agent decides and acts. It uses the model to plan steps, call tools, look at the results and continue until the task is done.

An Agent in Bedrock is made of:

Part	What it is	SRE Assistant example
Foundation model	The reasoning brain	Claude
Instructions	The agent’s role and rules	“You are an SRE assistant. Be concise. Never close incidents yourself.”
Action Groups	Tools it can call, usually backed by a Lambda function and an OpenAPI schema	getOpenCAPAs(service), getRecentDeployments(service)
Knowledge Base(s)	Documents it can search	Runbooks and RCA docs
Guardrails (optional)	Safety filters	Block secrets from appearing in answers

Example flow: “Payments service is erroring. What changed recently and what do our runbooks say?”

The agent reasons: “I need recent changes and runbook guidance.”
It calls the action getRecentDeployments("payments"), which hits your Change Tracker API.
It queries the Knowledge Base for the payments error runbook.
It combines both and answers: “A deployment went out 20 minutes ago. Runbook step 1 says to check rollback criteria...”

This loop of think, act, observe, repeat is why an agent can handle multi-step requests a plain model cannot.

6. Session

A session is one continuous conversation with an agent.

When you call the agent you pass a sessionId (any string you choose, e.g. a UUID):

Same sessionId → the agent treats the calls as one conversation and remembers what was said earlier.
New sessionId → a fresh conversation with a clean slate.

Example:


sessionId = "abc-123"
You:   "Show open CAPAs for payments"
Agent: "There are 4 open..."
You:   "Which one is oldest?"     ← works, same session "abc-123"

sessionId = "xyz-789" (new)
You:   "Which one is oldest?"     ← agent: "Oldest what?" (no history)

Sessions expire after an idle timeout, and you can also set session attributes: key-value data you pass along (like userId=priya, team=SRE) that tools can use.

Isolation: one user’s session is not visible to another. The newer AgentCore Runtime also provides session isolation for thousands of concurrent users. 
aws-news

7. Memory

There are two kinds, and people often mix them up.

Short-term memory (within a session)

The conversation history of the current session. It is what lets “Which one is oldest?” work above. It goes away when the session ends.

Long-term memory (across sessions)

Facts that persist after the session ends, so the agent doesn’t start from zero next time. In the classic Bedrock Agents, you enable this with a memoryId that identifies the user. The agent stores summaries of past sessions under that ID.

Example:

Monday session: you tell the assistant “I mainly care about the payments and checkout services.”
Wednesday, new session, same memoryId: “Any new alerts for me?” The agent already knows to focus on payments and checkout.

In AgentCore Memory, the newer design, short-term memory is organized by actor_id and session_id, while long-term memory extracts and keeps user preferences across sessions using automated memory strategies. So there are three IDs to know apart: 
aws-news

ID	Meaning
sessionId	Which conversation
memoryId / actor_id	Which user the long-term memory belongs to
agentId	Which agent
8. Agent versions and Alias ID

This is the part that confuses most people, so here is the analogy first. It’s like Git branches and releases.

DRAFT: the working copy where you edit and test the agent. Always changing.
Version (1, 2, 3...): when you’re happy, you create a version. It is an immutable snapshot of the agent at that moment (instructions, model, action groups, knowledge bases). It can never be edited.
Alias: a named pointer to a version, such as prod, staging or test. Each alias has an Alias ID.

You don’t call the agent by version number. You call it by agentId + agentAliasId.

Why this is useful:


Alias "prod"    → points to Version 2   (what your portal uses)
Alias "staging" → points to Version 3   (new prompt being tested)

Once Version 3 is tested, you repoint prod to Version 3. Your application code doesn’t change at all because it still calls the same alias ID. If Version 3 is bad, you point prod back to Version 2 and that is your instant rollback.

Bedrock also gives you a built-in test alias (TSTALIASID) that points to DRAFT, for quick testing.

This is similar to how you deploy with Helm and Kubernetes. An alias is like a stable Service name, and versions are like image tags behind it.

9. Putting it all together

                User (via your SM Portal UI)
                          │  sessionId, memoryId
                          ▼
              ┌─────────────────────────┐
              │  Agent (agentId + alias)│
              │  Model: Claude          │
              │  Instructions           │
              └───┬─────────────┬───────┘
                  │             │
        Action Group         Knowledge Base
        (Lambda → your       (S3 docs → chunks →
         Change/CAPA APIs)    embeddings → Vector DB)
                  │             │
                  └──────┬──────┘
                         ▼
                 Final answer to user
         (Guardrails check input and output)

Calling it from code (Python):

python
import boto3
client = boto3.client("bedrock-agent-runtime")

response = client.invoke_agent(
    agentId="ABCDEFGHIJ",
    agentAliasId="PRODALIAS1",     # which version, via alias
    sessionId="abc-123",           # which conversation
    memoryId="user-priya",         # long-term memory owner
    inputText="What changed in payments recently?",
)
10. Quick glossary
Term	One-line meaning
Model	The LLM that generates text
Token	A small chunk of text, the unit models count
Context window	Max tokens the model can see in one request
Embedding	Numbers representing the meaning of text
Vector DB	Database that finds similar embeddings
Knowledge Base	Managed pipeline: docs → chunks → vectors → search
RAG	Retrieve relevant info, then generate an answer with it
Agent	Model plus instructions plus tools plus knowledge, able to act
Action Group	The tools (APIs/Lambdas) an agent can call
Session	One conversation, identified by sessionId
Short-term memory	History within a session
Long-term memory	Facts kept across sessions (memoryId)
Version	Frozen snapshot of an agent
Alias	Named pointer to a version (e.g. prod), with an Alias ID
Guardrails	Content and safety filters
11. Bedrock AgentCore (the newer layer)

You will see this name a lot in 2026. Classic Bedrock Agents is the simple, configuration-driven way to build an agent. AgentCore is a set of infrastructure services for running agents you build with your own code or frameworks. According to an AWS Builder Center overview, AgentCore Memory provides persistent context so agents keep state across sessions instead of starting from zero on each call. AWS also describes a harness that runs the orchestration loop, executes tools, manages context, persists state across turns, recovers from failures and isolates sessions.
