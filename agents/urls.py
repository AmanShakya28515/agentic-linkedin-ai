from django.urls import path
from .views import (
    generate_linkedin_post,
    generate_linkedin_post_stream,
    start_workflow,
    resume_workflow,
    chat_view,
    message,
    message_stream,
    campaign,
    upload_document,
    remember_this,
)

urlpatterns = [
    # Phase 1 & 2 — Postman endpoints
    path("generate-linkedin-post/", generate_linkedin_post),
    path("generate-linkedin-post/stream/", generate_linkedin_post_stream),
    path("start/", start_workflow),
    path("resume/", resume_workflow),

    # Phase 3 — Chat UI
    path("chat/", chat_view),

    # Phase 4 — Orchestrator: single endpoint for all user messages
    path("message/", message),

    # Phase 6 — Streaming: new topic uses SSE, feedback/approval still use /message/
    path("message/stream/", message_stream),

    # Phase 15 — Multi-Post Campaign
    path("campaign/", campaign),

    # Phase 16 — Document Upload to RAG
    path("upload/", upload_document),
    path("remember/", remember_this),
]
