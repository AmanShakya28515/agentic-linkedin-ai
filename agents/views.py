import json
import uuid
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from .agent_flow import run_agent_flow, stream_agent_flow, start_agent_flow, resume_agent_flow
from .skills.orchestrator_skill import orchestrator_skill  # Phase 4
from .skills.research_skill import research_skill          # Phase 6
from .skills.hook_skill import hook_skill                  # Phase 6
from .skills.writing_skill import writing_skill_stream     # Phase 6
from .supervisor_flow import start_supervisor_flow, resume_supervisor_flow, start_supervisor_flow_with_draft  # Phase 7
from .sub_agents.campaign_agent import run_campaign_agent          # Phase 15
from .skills.document_skill import store_document, store_plain_text, list_documents  # Phase 16


# Phase 3 - Chat UI: serves the chat interface page
def chat_view(request):
    return render(request, "agents/chat.html")


@csrf_exempt
def generate_linkedin_post(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    topic = data.get("topic")
    if not topic:
        return JsonResponse({"error": "Topic is required"}, status=400)

    result = run_agent_flow(topic)

    return JsonResponse({
        "topic": result["topic"],
        "research": result["research"],
        "hook": result["hook"],
        "draft": result["draft"],
        "review_decision": result["review_decision"],
        "review_feedback": result["review_feedback"],
        "revision_count": result["revision_count"],
        "final_post": result["final_post"],
    })


@csrf_exempt
def generate_linkedin_post_stream(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    topic = data.get("topic")
    if not topic:
        return JsonResponse({"error": "Topic is required"}, status=400)

    steps = stream_agent_flow(topic)
    return JsonResponse({"topic": topic, "steps": steps})


@csrf_exempt
def start_workflow(request):
    """
    Step 1 of HITL: Start the workflow.
    Runs research → hook → writer, then PAUSES for human review.
    Returns thread_id and the draft for the human to review.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    topic = data.get("topic")
    if not topic:
        return JsonResponse({"error": "Topic is required"}, status=400)

    thread_id = str(uuid.uuid4())

    try:
        result = start_agent_flow(topic, thread_id)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({
        "thread_id": thread_id,
        "status": "waiting_for_human",
        "draft": result.get("draft", ""),
        "message": "Review the draft and POST to /resume/ with your decision.",
    })


@csrf_exempt
def resume_workflow(request):
    """
    Step 2 of HITL: Human submits their review decision.
    Resumes the paused workflow — runs reviewer → final output.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    thread_id = data.get("thread_id")
    if not thread_id:
        return JsonResponse({"error": "thread_id is required"}, status=400)

    approved = data.get("approved", True)
    feedback = data.get("feedback", "")

    result = resume_agent_flow(thread_id, {"approved": approved, "feedback": feedback})

    # If final_post is set the graph completed — otherwise it paused again at human_review
    if result.get("final_post"):
        return JsonResponse({
            "thread_id": thread_id,
            "status": "completed",
            "final_post": result.get("final_post", ""),
        })
    else:
        return JsonResponse({
            "thread_id": thread_id,
            "status": "waiting_for_human",
            "draft": result.get("draft", ""),
        })


# =============================================================================
# PHASE 4 — Orchestrator Agent
# Single endpoint that handles ALL user messages.
# Orchestrator classifies intent → routes to correct action automatically.
# Phase 3 used manual buttons (approved/feedback/new_topic).
# Phase 4 removes buttons — one input box, AI decides what to do.
# =============================================================================
@csrf_exempt
def message(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    user_message = data.get("message", "").strip()
    thread_id    = data.get("thread_id")       # None if no workflow running yet

    if not user_message:
        return JsonResponse({"error": "message is required"}, status=400)

    # ── No active workflow → this must be a new topic ──────────────────────
    # Phase 4-6: start_agent_flow(user_message, new_thread_id)
    # Phase 7:   start_supervisor_flow — delegates to sub-agents internally
    if not thread_id:
        new_thread_id = str(uuid.uuid4())
        try:
            result = start_supervisor_flow(user_message, new_thread_id)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

        return JsonResponse({
            "thread_id": new_thread_id,
            "status": "waiting_for_human",
            "intent": "new_topic",
            "draft": result.get("draft", ""),
        })

    # ── Active workflow → orchestrator classifies intent ────────────────────
    current_state = "waiting_for_human_review"
    intent = orchestrator_skill(user_message, current_state)

    if intent == "new_topic":
        new_thread_id = str(uuid.uuid4())
        try:
            result = start_supervisor_flow(user_message, new_thread_id)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

        return JsonResponse({
            "thread_id": new_thread_id,
            "status": "waiting_for_human",
            "intent": "new_topic",
            "draft": result.get("draft", ""),
        })

    elif intent == "approved":
        # Phase 4-6: resume_agent_flow(...)
        # Phase 7:   resume_supervisor_flow — resumes supervisor checkpoint
        result = resume_supervisor_flow(thread_id, {"approved": True, "feedback": ""})
        return JsonResponse({
            "thread_id": thread_id,
            "status": "completed",
            "intent": "approved",
            "final_post": result.get("final_post", ""),
        })

    else:  # feedback
        result = resume_supervisor_flow(thread_id, {"approved": False, "feedback": user_message})
        if result.get("final_post"):
            return JsonResponse({
                "thread_id": thread_id,
                "status": "completed",
                "intent": "feedback",
                "final_post": result.get("final_post", ""),
            })
        return JsonResponse({
            "thread_id": thread_id,
            "status": "waiting_for_human",
            "intent": "feedback",
            "draft": result.get("draft", ""),
        })


# =============================================================================
# PHASE 6 — Streaming
# New topic flow: streams status updates + writer tokens via SSE.
# After streaming, saves state to LangGraph (at human_review) for HITL.
# Feedback/approval still use the regular /message/ endpoint above.
# =============================================================================
@csrf_exempt
def message_stream(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    topic = data.get("message", "").strip()
    if not topic:
        return JsonResponse({"error": "message is required"}, status=400)

    thread_id = str(uuid.uuid4())

    def event_stream():
        yield f'data: {json.dumps({"type": "status", "text": "🔍 Researching your topic..."})}\n\n'

        research = research_skill(topic)

        yield f'data: {json.dumps({"type": "status", "text": "🪝 Generating hooks..."})}\n\n'

        hook = hook_skill(research)

        yield f'data: {json.dumps({"type": "status", "text": "✍️ Writing your draft..."})}\n\n'

        # Stream writer tokens to browser word by word
        draft_tokens = []
        for token in writing_skill_stream(topic=topic, research=research, hook=hook):
            draft_tokens.append(token)
            yield f'data: {json.dumps({"type": "token", "text": token})}\n\n'

        draft = "".join(draft_tokens)

        # Phase 6 original: start_agent_flow_with_draft (agent_flow MemorySaver)
        # Phase 7 fix: use supervisor MemorySaver so feedback/approval find the checkpoint
        start_supervisor_flow_with_draft(topic, research, hook, draft, thread_id)

        yield f'data: {json.dumps({"type": "done", "thread_id": thread_id})}\n\n'

    response = StreamingHttpResponse(event_stream(), content_type="text/event-stream")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"
    return response


# =============================================================================
# PHASE 15 — Multi-Post Campaign
# Generates a full content campaign (multiple posts across multiple days).
# No HITL — returns all posts at once.
# =============================================================================
@csrf_exempt
def campaign(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    topic    = data.get("topic", "").strip()
    num_days = int(data.get("num_days", 5))

    if not topic:
        return JsonResponse({"error": "topic is required"}, status=400)

    try:
        result = run_campaign_agent(topic, num_days)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse(result)


# =============================================================================
# PHASE 16 — Document Upload to RAG
# Supports: PDF, TXT, XLSX — text extracted → chunked → stored in ChromaDB
# =============================================================================
ALLOWED_EXTENSIONS = (".pdf", ".txt", ".xlsx", ".xls")

@csrf_exempt
def upload_document(request):
    if request.method == "GET":
        return JsonResponse({"documents": list_documents()})

    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    uploaded_file = request.FILES["file"]
    filename = uploaded_file.name
    ext = "." + filename.lower().split(".")[-1]

    if ext not in ALLOWED_EXTENSIONS:
        return JsonResponse({
            "error": f"Unsupported file type. Allowed: PDF, TXT, XLSX"
        }, status=400)

    import tempfile, os as _os
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        for chunk in uploaded_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        num_chunks = store_document(tmp_path, filename)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
    finally:
        _os.unlink(tmp_path)

    return JsonResponse({
        "message": f"✅ '{filename}' uploaded — {num_chunks} chunks added to knowledge base.",
        "filename": filename,
        "chunks": num_chunks,
    })


# =============================================================================
# PHASE 16 — "Remember This" — store plain text directly from chat
# User types: "remember this: our company is Plutonic Tech..."
# =============================================================================
@csrf_exempt
def remember_this(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST allowed"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    text = data.get("text", "").strip()
    if not text:
        return JsonResponse({"error": "text is required"}, status=400)

    try:
        num_chunks = store_plain_text(text, label="user_note")
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({
        "message": f"✅ Remembered — {num_chunks} chunks added to knowledge base.",
        "chunks": num_chunks,
    })
