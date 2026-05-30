import os
import chromadb

# Phase 11 — RAG (Retrieval Augmented Generation)
# Stores approved posts + viral style examples in ChromaDB.
# Before writing, retrieves similar posts to teach the LLM hook structure,
# pacing, storytelling rhythm, and emotional tension — without hardcoding everything.
#
# Flow:
#   STORE:    approved post / viral example → vector embedding → ChromaDB
#   RETRIEVE: new topic → find similar posts → inject style into writing prompt

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "chroma_db")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="linkedin_posts")

# ── Viral style examples seeded on first run ──────────────────────────────────
VIRAL_EXAMPLES = [
    {
        "id": "viral_1",
        "topic": "discipline and consistency",
        "post": """Nobody wakes up talented.

They wake up early.

Messi touched a ball 500 times a day at age 8.
Kobe studied defender tendencies before every game.
Ronaldo trains when teammates go home.

The gap between good and great isn't talent.
It's the hours no one sees.

What are you doing when no one is watching?"""
    },
    {
        "id": "viral_2",
        "topic": "AI and future of work",
        "post": """AI won't take your job.

Someone using AI will.

That's not a threat. It's a fact.

In 2023, companies started replacing entire departments.
Not with robots. With one person who knew the right prompts.

The skill isn't coding.
The skill is knowing what to ask.

Are you learning to use AI — or waiting to be replaced by someone who does?"""
    },
    {
        "id": "viral_3",
        "topic": "startup failure and resilience",
        "post": """My startup failed in 11 months.

I had 3 investors, 8 employees, and zero revenue.

Here's what I learned that MBA programs won't teach you:

1. Speed beats perfection. Always.
2. Your first customer matters more than your pitch deck.
3. Cash flow is oxygen. Everything else is a nice-to-have.

Failure isn't the opposite of success.
It's part of the route.

What's a failure that taught you more than any win?"""
    },
    {
        "id": "viral_4",
        "topic": "leadership and management",
        "post": """The best manager I ever had said something I'll never forget:

"My job is to make you not need me."

Most managers do the opposite.
They create dependency. They hoard information. They micromanage.

Real leadership is building people who outgrow you.

The measure of a great leader isn't how many followers they have.
It's how many leaders they created.

Who made you better by believing in you?"""
    },
    {
        "id": "viral_5",
        "topic": "productivity and focus",
        "post": """You don't have a productivity problem.

You have a priority problem.

Everyone has 24 hours.
Elon. Beyoncé. You.

The difference isn't time. It's what they say no to.

Stop optimizing your schedule.
Start eliminating what doesn't belong on it.

One question: What's on your calendar that shouldn't be?"""
    },
]


def _seed_examples():
    """Seed viral examples once — skip if already present."""
    existing_ids = collection.get()["ids"]
    for example in VIRAL_EXAMPLES:
        if example["id"] not in existing_ids:
            collection.add(
                documents=[example["post"]],
                metadatas=[{"topic": example["topic"], "type": "viral_example"}],
                ids=[example["id"]],
            )


# Seed on import — runs once, skips if already seeded
_seed_examples()


def store_post(topic: str, post: str):
    """STORE: save an approved user post to the vector database."""
    existing_ids = collection.get()["ids"]
    doc_id = f"user_post_{len(existing_ids) + 1}"
    collection.add(
        documents=[post],
        metadatas=[{"topic": topic, "type": "user_post"}],
        ids=[doc_id],
    )


def retrieve_similar_posts(topic: str, n_results: int = 2) -> str:
    """RETRIEVE: find similar posts — viral examples + past user posts + uploaded documents."""
    context_parts = []

    # Search linkedin_posts collection (viral examples + user posts)
    total = len(collection.get()["ids"])
    if total > 0:
        results = collection.query(
            query_texts=[topic],
            n_results=min(n_results, total),
        )
        posts = results["documents"][0]
        metadatas = results["metadatas"][0]

        if posts:
            posts_context = "Study these LinkedIn posts for hook structure, pacing, and storytelling style:\n\n"
            for meta, post in zip(metadatas, posts):
                label = "Viral example" if meta.get("type") == "viral_example" else "Your past post"
                posts_context += f"{label} (topic: {meta['topic']}):\n{post}\n\n"
            context_parts.append(posts_context.strip())

    # Phase 16 — also search uploaded documents
    from .document_skill import retrieve_from_documents
    doc_context = retrieve_from_documents(topic)
    if doc_context:
        context_parts.append(doc_context)

    return "\n\n".join(context_parts) if context_parts else ""
