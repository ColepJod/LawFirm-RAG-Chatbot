const STORAGE = {
  conversation: "lawhelp.conversationId",
  adminToken: "lawhelp.adminToken",
};

const NODE_LABELS = {
  route: "Routing question",
  condense_question: "Understanding the conversation",
  retrieve: "Searching documents",
  grade_documents: "Checking relevance",
  rerank: "Ranking sources",
  rewrite_query: "Refining the search",
  generate: "Writing the answer",
  check_grounding: "Verifying citations",
  direct_answer: "Composing a reply",
  log_trace: "Finishing",
};

const el = {
  messages: document.getElementById("messages"),
  emptyState: document.getElementById("emptyState"),
  composer: document.getElementById("composer"),
  query: document.getElementById("query"),
  send: document.getElementById("send"),
  newChat: document.getElementById("newChat"),
  health: document.getElementById("health"),
  sources: document.getElementById("sources"),
  sourceCount: document.getElementById("sourceCount"),
  documents: document.getElementById("documents"),
  refreshDocs: document.getElementById("refreshDocs"),
  adminToken: document.getElementById("adminToken"),
  uploadForm: document.getElementById("uploadForm"),
  uploadFile: document.getElementById("uploadFile"),
  directoryForm: document.getElementById("directoryForm"),
  directoryPath: document.getElementById("directoryPath"),
  jobStatus: document.getElementById("jobStatus"),
};

let conversationId = localStorage.getItem(STORAGE.conversation);
if (!conversationId) {
  conversationId = newConversationId();
  localStorage.setItem(STORAGE.conversation, conversationId);
}
let adminToken = sessionStorage.getItem(STORAGE.adminToken) || "";
let busy = false;

function newConversationId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `chat-${Date.now()}-${Math.random().toString(16).slice(2, 10)}`;
}

function adminHeaders(extra = {}) {
  return adminToken ? { ...extra, "X-Admin-Token": adminToken } : extra;
}

function setHealth(text, kind = "muted") {
  el.health.textContent = text;
  el.health.className = `pill pill-${kind}`;
}

function setJobStatus(text, kind = "") {
  el.jobStatus.textContent = text;
  el.jobStatus.className = `job-status ${kind}`.trim();
}

function scrollToBottom() {
  el.messages.scrollTop = el.messages.scrollHeight;
}

function hideEmptyState() {
  el.emptyState.hidden = true;
}

function addMessage(text, role, { pending = false, error = false } = {}) {
  hideEmptyState();
  const node = document.createElement("div");
  node.className = `message ${role}${pending ? " pending" : ""}${error ? " error" : ""}`;
  node.textContent = text;
  el.messages.appendChild(node);
  scrollToBottom();
  return node;
}

function addMeta(messageNode, payload) {
  const meta = document.createElement("div");
  meta.className = "message-meta";
  const parts = [];
  if (payload.grounding_verdict) parts.push(`grounding: ${payload.grounding_verdict}`);
  if (typeof payload.latency_ms === "number") parts.push(`${Math.round(payload.latency_ms)} ms`);
  if (payload.thread_id) parts.push(`conversation ${payload.thread_id.slice(0, 8)}`);
  if (!parts.length) return;
  meta.textContent = parts.join(" · ");
  messageNode.appendChild(meta);
}

function addWarnings(messageNode, warnings) {
  if (!warnings || !warnings.length) return;
  const list = document.createElement("div");
  list.className = "warning-list";
  list.textContent = warnings.join(" ");
  messageNode.appendChild(list);
}

function relevancePercent(score) {
  // Cross-encoder scores are raw logits; map them to 0-100 for display.
  const probability = 1 / (1 + Math.exp(-score));
  return `${Math.round(probability * 100)}%`;
}

function renderSources(sources) {
  el.sources.replaceChildren();
  el.sourceCount.textContent = String(sources ? sources.length : 0);
  if (!sources || !sources.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No sources were returned for this answer.";
    el.sources.appendChild(empty);
    return;
  }

  for (const source of sources) {
    const item = document.createElement("article");
    item.className = "source-item";

    const header = document.createElement("header");
    const title = document.createElement("span");
    title.className = "source-title";
    title.textContent = `[${source.id}] ${source.source_file}`;
    header.appendChild(title);
    if (typeof source.score === "number") {
      const score = document.createElement("span");
      score.className = "pill pill-muted";
      score.textContent = `${relevancePercent(source.score)} match`;
      score.title = `Cross-encoder score: ${source.score.toFixed(2)}`;
      header.appendChild(score);
    }
    item.appendChild(header);

    const sub = document.createElement("div");
    sub.className = "source-sub";
    const page = source.page == null ? "page n/a" : `page ${source.page}`;
    sub.textContent = `${source.section || "document"} · ${page}`;
    item.appendChild(sub);

    if (source.snippet) {
      const snippet = document.createElement("p");
      snippet.className = "snippet";
      snippet.textContent = source.snippet;
      item.appendChild(snippet);
    }
    el.sources.appendChild(item);
  }
}

function parseEventBlock(block) {
  let event = "message";
  const dataLines = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
  }
  let data = null;
  if (dataLines.length) {
    try {
      data = JSON.parse(dataLines.join("\n"));
    } catch {
      data = dataLines.join("\n");
    }
  }
  return { event, data };
}

function errorMessageFromBody(body, fallback) {
  if (!body) return fallback;
  const detail = body.detail ?? body;
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object") {
    if (detail.message) return detail.message;
    if (detail.reasons) return detail.reasons.join("; ");
  }
  return fallback;
}

async function request(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: adminHeaders(options.headers || {}),
  });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const error = new Error(errorMessageFromBody(body, `Request failed (${response.status})`));
    error.status = response.status;
    throw error;
  }
  return response;
}

async function sendQuery(query) {
  if (busy || !query.trim()) return;
  busy = true;
  el.send.disabled = true;
  el.query.disabled = true;
  addMessage(query, "user");
  el.query.value = "";
  resizeComposer();

  const pendingNode = addMessage("Routing question…", "assistant", { pending: true });

  try {
    const response = await fetch("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, conversation_id: conversationId }),
    });
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new Error(errorMessageFromBody(body, `Request failed (${response.status})`));
    }
    if (!response.body) throw new Error("Streaming is not supported by this browser.");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let finished = null;
    let streamError = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        const block = buffer.slice(0, boundary).trim();
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf("\n\n");
        if (!block) continue;

        const { event, data } = parseEventBlock(block);
        if (event === "progress" && data) {
          pendingNode.textContent = `${NODE_LABELS[data.node] || "Working"}…`;
        } else if (event === "done" && data) {
          finished = data;
        } else if (event === "error") {
          streamError = errorMessageFromBody(data, "The request failed.");
        }
      }
    }

    if (streamError) throw new Error(streamError);
    if (!finished) throw new Error("The server did not return an answer.");

    pendingNode.classList.remove("pending");
    pendingNode.textContent = finished.answer || "No answer returned.";
    addWarnings(pendingNode, finished.warnings);
    addMeta(pendingNode, finished);
    renderSources(finished.sources);
  } catch (error) {
    pendingNode.remove();
    addMessage(`Error: ${error.message}`, "assistant", { error: true });
  } finally {
    busy = false;
    el.send.disabled = false;
    el.query.disabled = false;
    el.query.focus();
    scrollToBottom();
  }
}

async function refreshHealth() {
  try {
    const response = await request("/health");
    const body = await response.json();
    const chunks = body.checks?.index?.chunk_count;
    const suffix = typeof chunks === "number" ? ` · ${chunks} chunks` : "";
    setHealth(`${body.status === "ok" ? "Ready" : "Degraded"}${suffix}`, body.status === "ok" ? "ok" : "warn");
  } catch {
    setHealth("Offline", "bad");
  }
}

function documentPath(sourceFile) {
  return sourceFile
    .split("/")
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

async function refreshDocuments() {
  try {
    const response = await request("/documents");
    const body = await response.json();
    el.documents.replaceChildren();
    if (!body.documents?.length) {
      const empty = document.createElement("p");
      empty.className = "muted";
      empty.textContent = "No indexed documents. Upload a file or index a directory.";
      el.documents.appendChild(empty);
      return;
    }
    for (const documentInfo of body.documents) {
      const item = document.createElement("article");
      item.className = "document-item";

      const header = document.createElement("header");
      const name = document.createElement("span");
      name.className = "document-name";
      name.textContent = documentInfo.source_file;
      header.appendChild(name);

      const button = document.createElement("button");
      button.type = "button";
      button.className = "button-danger";
      button.textContent = "Delete";
      button.addEventListener("click", () => deleteDocument(documentInfo.source_file));
      header.appendChild(button);
      item.appendChild(header);

      const sub = document.createElement("div");
      sub.className = "source-sub";
      sub.textContent = `${documentInfo.chunks} chunk${documentInfo.chunks === 1 ? "" : "s"}`;
      item.appendChild(sub);
      el.documents.appendChild(item);
    }
  } catch (error) {
    el.documents.replaceChildren();
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent =
      error.status === 401 ? "Enter an admin token to view the knowledge base." : error.message;
    el.documents.appendChild(empty);
  }
}

async function deleteDocument(sourceFile) {
  if (!window.confirm(`Delete all indexed chunks for ${sourceFile}?`)) return;
  try {
    await request(`/documents/${documentPath(sourceFile)}`, { method: "DELETE" });
    setJobStatus(`Deleted ${sourceFile}.`, "ok");
    await refreshDocuments();
    await refreshHealth();
    setSourcesNotice("Sources refreshed. Ask a question again.");
  } catch (error) {
    setJobStatus(`Delete failed: ${error.message}`, "bad");
  }
}

function setSourcesNotice(text) {
  el.sources.replaceChildren();
  el.sourceCount.textContent = "0";
  const empty = document.createElement("p");
  empty.className = "muted";
  empty.textContent = text;
  el.sources.appendChild(empty);
}

async function pollJob(jobId) {
  const started = Date.now();
  while (Date.now() - started < 180000) {
    await new Promise((resolve) => setTimeout(resolve, 1200));
    try {
      const response = await request(`/jobs/${jobId}`);
      const job = await response.json();
      if (job.status === "succeeded") {
        setJobStatus(`Indexed ${job.documents} document(s), ${job.chunks} chunk(s).`, "ok");
        await refreshDocuments();
        await refreshHealth();
        return;
      }
      if (job.status === "failed") {
        setJobStatus(`Indexing failed: ${job.error || "unknown error"}`, "bad");
        return;
      }
      setJobStatus(`Indexing ${job.status}…`);
    } catch (error) {
      setJobStatus(`Job status unavailable: ${error.message}`, "bad");
      return;
    }
  }
  setJobStatus("Indexing is taking longer than expected. Check again later.", "bad");
}

async function submitIngestion(formData) {
  try {
    setJobStatus("Uploading…");
    const response = await request("/ingest", { method: "POST", body: formData });
    const body = await response.json();
    setJobStatus(body.message || "Queued for indexing.");
    if (body.job_id) await pollJob(body.job_id);
  } catch (error) {
    setJobStatus(
      error.status === 401 ? "Admin token required or invalid." : error.message,
      "bad",
    );
  }
}

function resizeComposer() {
  el.query.style.height = "auto";
  el.query.style.height = `${Math.min(el.query.scrollHeight, 160)}px`;
}

el.composer.addEventListener("submit", (event) => {
  event.preventDefault();
  sendQuery(el.query.value);
});

el.query.addEventListener("input", resizeComposer);
el.query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    el.composer.requestSubmit();
  }
});

el.newChat.addEventListener("click", () => {
  conversationId = newConversationId();
  localStorage.setItem(STORAGE.conversation, conversationId);
  el.messages.replaceChildren(el.emptyState);
  el.emptyState.hidden = false;
  setSourcesNotice("Ask a question to see the documents used for the answer.");
  el.query.focus();
});

for (const button of document.querySelectorAll(".suggestion")) {
  button.addEventListener("click", () => {
    el.query.value = button.dataset.question || "";
    el.composer.requestSubmit();
  });
}

el.adminToken.value = adminToken;
el.adminToken.addEventListener("change", () => {
  adminToken = el.adminToken.value.trim();
  sessionStorage.setItem(STORAGE.adminToken, adminToken);
  refreshDocuments();
});

el.refreshDocs.addEventListener("click", refreshDocuments);

el.uploadForm.addEventListener("submit", (event) => {
  event.preventDefault();
  if (!el.uploadFile.files?.length) {
    setJobStatus("Choose a file first.", "bad");
    return;
  }
  const formData = new FormData();
  formData.append("file", el.uploadFile.files[0]);
  submitIngestion(formData).finally(() => {
    el.uploadFile.value = "";
  });
});

el.directoryForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const directory = el.directoryPath.value.trim();
  if (!directory) {
    setJobStatus("Enter a directory path.", "bad");
    return;
  }
  const formData = new FormData();
  formData.append("directory", directory);
  submitIngestion(formData);
});

refreshHealth();
refreshDocuments();
setSourcesNotice("Ask a question to see the documents used for the answer.");
el.query.focus();
