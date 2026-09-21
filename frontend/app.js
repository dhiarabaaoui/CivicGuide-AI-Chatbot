const HISTORY_STORAGE_KEY = "civicguide-ai-conversations-v1";
const MAX_SAVED_CONVERSATIONS = 12;

const state = {
  history: [],
  domain: "",
  busy: false,
  pipelineTimer: null,
  sources: [],
  demoToken: 0,
  conversations: [],
  currentConversationId: null,
};

const elements = {
  sidebar: document.querySelector("#sidebar"),
  newChat: document.querySelector("#newChat"),
  historyList: document.querySelector("#historyList"),
  menuButton: document.querySelector("#menuButton"),
  domainLabel: document.querySelector("#domainLabel"),
  demoButton: document.querySelector("#demoButton"),
  openSources: document.querySelector("#openSources"),
  sourceCount: document.querySelector("#sourceCount"),
  welcome: document.querySelector("#welcome"),
  messages: document.querySelector("#messages"),
  pipelineProgress: document.querySelector("#pipelineProgress"),
  pipelineLabel: document.querySelector("#pipelineLabel"),
  composer: document.querySelector("#composer"),
  messageInput: document.querySelector("#messageInput"),
  sendButton: document.querySelector("#sendButton"),
  evidencePanel: document.querySelector("#evidencePanel"),
  closeEvidence: document.querySelector("#closeEvidence"),
  drawerCount: document.querySelector("#drawerCount"),
  sourceList: document.querySelector("#sourceList"),
  mobileOverlay: document.querySelector("#mobileOverlay"),
};

const domainNames = {
  "": "Assistant général",
  dmv: "DMV",
  ssa: "Social Security",
  va: "Veterans Affairs",
  studentaid: "Student Aid",
};

function escapeHtml(value = "") {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function safeExternalUrl(value) {
  if (!value) return "";
  try {
    const url = new URL(value);
    return ["http:", "https:"].includes(url.protocol) ? url.href : "";
  } catch {
    return "";
  }
}

function autoResize() {
  elements.messageInput.style.height = "auto";
  elements.messageInput.style.height = `${Math.min(elements.messageInput.scrollHeight, 150)}px`;
}

function updateSendState() {
  elements.sendButton.disabled = state.busy || !elements.messageInput.value.trim();
}

function delay(milliseconds) {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
}

function loadSavedConversations() {
  try {
    const saved = JSON.parse(localStorage.getItem(HISTORY_STORAGE_KEY) || "[]");
    state.conversations = Array.isArray(saved) ? saved.slice(0, MAX_SAVED_CONVERSATIONS) : [];
  } catch {
    state.conversations = [];
  }
}

function conversationTitle() {
  const firstQuestion = state.history.find((turn) => turn.role === "user")?.content || "Nouvelle conversation";
  return firstQuestion.length > 38 ? `${firstQuestion.slice(0, 38).trim()}…` : firstQuestion;
}

function relativeDate(timestamp) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "Récemment";
  return new Intl.DateTimeFormat("fr", { day: "2-digit", month: "short" }).format(date);
}

function renderConversationHistory() {
  if (!state.conversations.length) {
    elements.historyList.innerHTML = '<p class="history-empty">Vos conversations récentes apparaîtront ici.</p>';
    return;
  }

  elements.historyList.innerHTML = state.conversations.map((conversation) => `
    <button class="history-item${conversation.id === state.currentConversationId ? " active" : ""}" type="button" data-conversation-id="${escapeHtml(conversation.id)}">
      <strong>${escapeHtml(conversation.title || "Conversation")}</strong>
      <span>${escapeHtml(domainNames[conversation.domain] || "Assistant général")} · ${escapeHtml(relativeDate(conversation.updatedAt))}</span>
    </button>
  `).join("");
}

function persistCurrentConversation() {
  if (!state.history.length) return;
  if (!state.currentConversationId) {
    state.currentConversationId = `conversation-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  }

  const conversation = {
    id: state.currentConversationId,
    title: conversationTitle(),
    domain: state.domain,
    history: state.history,
    sources: state.sources,
    updatedAt: new Date().toISOString(),
  };
  state.conversations = [
    conversation,
    ...state.conversations.filter((item) => item.id !== conversation.id),
  ].slice(0, MAX_SAVED_CONVERSATIONS);

  try {
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(state.conversations));
  } catch {
    // Le chatbot continue à fonctionner si le stockage local est indisponible.
  }
  renderConversationHistory();
}

function hideWelcome() {
  elements.welcome.classList.add("hidden");
}

function scrollConversation() {
  const stage = document.querySelector(".chat-stage");
  requestAnimationFrame(() => {
    stage.scrollTop = stage.scrollHeight;
  });
}

function closePanels() {
  elements.sidebar.classList.remove("open");
  elements.evidencePanel.classList.remove("open");
  elements.evidencePanel.setAttribute("aria-hidden", "true");
  elements.openSources.setAttribute("aria-expanded", "false");
  elements.mobileOverlay.classList.remove("visible");
}

function openSourceDrawer(focusId = "") {
  elements.sidebar.classList.remove("open");
  elements.evidencePanel.classList.add("open");
  elements.evidencePanel.setAttribute("aria-hidden", "false");
  elements.openSources.setAttribute("aria-expanded", "true");
  elements.mobileOverlay.classList.add("visible");

  if (focusId) {
    requestAnimationFrame(() => {
      const card = elements.sourceList.querySelector(`[data-evidence-id="${CSS.escape(focusId)}"]`);
      if (!card) return;
      card.classList.remove("highlight");
      void card.offsetWidth;
      card.classList.add("highlight");
      card.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }
}

function formatAnswer(answer, citations = []) {
  let formatted = escapeHtml(answer).replaceAll("\n", "<br>");
  citations.forEach((citation, index) => {
    const id = citation.evidence_id || `E${index + 1}`;
    const marker = new RegExp(`\\[${id.replace(/[.*+?^$\{\}()|[\]\\]/g, "\\$&")}\\]`, "g");
    const button = `<button class="citation" type="button" data-citation="${escapeHtml(id)}" aria-label="Voir la source ${index + 1}">${index + 1}</button>`;
    formatted = formatted.replace(marker, button);
  });

  const containsCitation = formatted.includes('data-citation="');
  if (!containsCitation && citations.length) {
    formatted += " " + citations
      .map((citation, index) => {
        const id = citation.evidence_id || `E${index + 1}`;
        return `<button class="citation" type="button" data-citation="${escapeHtml(id)}" aria-label="Voir la source ${index + 1}">${index + 1}</button>`;
      })
      .join(" ");
  }
  return formatted;
}

function appendMessage(role, content, citations = [], traceId = "") {
  hideWelcome();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  if (traceId) article.dataset.traceId = traceId;

  if (role === "assistant") {
    const feedback = traceId ? `
      <div class="message-feedback" aria-label="Évaluer cette réponse">
        <span>Cette réponse vous aide ?</span>
        <button type="button" data-feedback="helpful" aria-label="Réponse utile">Oui</button>
        <button type="button" data-feedback="not_helpful" aria-label="Réponse non utile">Non</button>
      </div>
    ` : "";
    article.innerHTML = `
      <div class="message-avatar" aria-hidden="true">CG</div>
      <div class="message-body"><p>${formatAnswer(content, citations)}</p>${feedback}</div>
    `;
  } else {
    article.innerHTML = `<div class="message-body"><p>${escapeHtml(content).replaceAll("\n", "<br>")}</p></div>`;
  }

  elements.messages.appendChild(article);
  scrollConversation();
  return article;
}

function appendTyping() {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.id = "typingMessage";
  article.innerHTML = `
    <div class="message-avatar" aria-hidden="true">CG</div>
    <div class="message-body">
      <div class="typing" aria-label="Réponse en cours">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  elements.messages.appendChild(article);
  scrollConversation();
}

function removeTyping() {
  document.querySelector("#typingMessage")?.remove();
}

function startPipeline() {
  const labels = [
    "Recherche des passages pertinents…",
    "Classement des meilleures sources…",
    "Construction de la réponse…",
    "Vérification des citations…",
  ];
  let step = 0;
  const dots = [...elements.pipelineProgress.querySelectorAll(".pipeline-dot")];
  elements.pipelineProgress.hidden = false;
  elements.pipelineLabel.textContent = labels[0];
  dots.forEach((dot, index) => dot.classList.toggle("active", index === 0));

  clearInterval(state.pipelineTimer);
  state.pipelineTimer = setInterval(() => {
    step = Math.min(step + 1, labels.length - 1);
    elements.pipelineLabel.textContent = labels[step];
    dots.forEach((dot, index) => dot.classList.toggle("active", index <= step));
  }, 850);
}

function stopPipeline() {
  clearInterval(state.pipelineTimer);
  state.pipelineTimer = null;
  elements.pipelineProgress.hidden = true;
}

function emptySourceMarkup() {
  return `
    <div class="empty-sources">
      <div aria-hidden="true">⌘</div>
      <strong>Aucune source pour le moment</strong>
      <p>Posez une question pour afficher les documents retrouvés.</p>
    </div>
  `;
}

function renderSources(citations = []) {
  state.sources = citations;
  const count = citations.length;
  elements.sourceCount.textContent = String(count);
  elements.drawerCount.textContent = String(count);

  if (!count) {
    elements.sourceList.innerHTML = emptySourceMarkup();
    return;
  }

  elements.sourceList.innerHTML = citations.map((citation, index) => {
    const id = citation.evidence_id || `E${index + 1}`;
    const title = citation.document_title || citation.title || "Document officiel";
    const section = citation.section_title || citation.document_id || "Passage documentaire";
    const quote = citation.evidence_quote || citation.text || "Passage utilisé pour justifier la réponse.";
    const url = safeExternalUrl(citation.url || citation.source_url || citation.source_document_id);
    const sourceLabel = url
      ? `<a href="${escapeHtml(url)}" target="_blank" rel="noreferrer">Ouvrir le document ↗</a>`
      : `<span>${escapeHtml(section)}</span>`;

    return `
      <article class="source-card" data-evidence-id="${escapeHtml(id)}">
        <div class="source-head">
          <span class="source-number">${index + 1}</span>
          <div class="source-title">
            <strong title="${escapeHtml(title)}">${escapeHtml(title)}</strong>
            ${sourceLabel}
          </div>
        </div>
        <blockquote>${escapeHtml(quote)}</blockquote>
        <div class="source-meta">
          <span>${escapeHtml(id)}</span>
          <span>${escapeHtml(citation.span_id || "")}</span>
        </div>
      </article>
    `;
  }).join("");
}

function resetConversation() {
  state.demoToken += 1;
  state.history = [];
  state.sources = [];
  state.currentConversationId = null;
  state.busy = false;
  clearInterval(state.pipelineTimer);
  elements.messages.innerHTML = "";
  elements.welcome.classList.remove("hidden");
  elements.pipelineProgress.hidden = true;
  elements.messageInput.value = "";
  autoResize();
  updateSendState();
  renderSources([]);
  renderConversationHistory();
  closePanels();
  elements.demoButton.disabled = false;
  elements.demoButton.textContent = "Voir une démo";
  elements.messageInput.focus();
}

function restoreConversation(conversationId) {
  const conversation = state.conversations.find((item) => item.id === conversationId);
  if (!conversation || state.busy) return;

  state.demoToken += 1;
  clearInterval(state.pipelineTimer);
  state.currentConversationId = conversation.id;
  state.domain = conversation.domain || "";
  state.history = Array.isArray(conversation.history) ? conversation.history : [];
  state.sources = Array.isArray(conversation.sources) ? conversation.sources : [];
  elements.messages.innerHTML = "";
  elements.welcome.classList.toggle("hidden", state.history.length > 0);
  state.history.forEach((turn) => appendMessage(
    turn.role,
    turn.content,
    turn.citations || [],
    turn.trace_id || "",
  ));
  renderSources(state.sources);
  document.querySelectorAll(".domain-item").forEach((item) => {
    item.classList.toggle("active", (item.dataset.domain || "") === state.domain);
  });
  elements.domainLabel.textContent = domainNames[state.domain] || "Assistant général";
  renderConversationHistory();
  closePanels();
}

function selectDomain(button) {
  state.domain = button.dataset.domain || "";
  document.querySelectorAll(".domain-item").forEach((item) => {
    item.classList.toggle("active", item === button);
  });
  elements.domainLabel.textContent = domainNames[state.domain] || "Assistant général";
  elements.sidebar.classList.remove("open");
  elements.mobileOverlay.classList.remove("visible");
}

async function runDemo() {
  resetConversation();
  const demoToken = state.demoToken;
  const question = "Quels documents faut-il pour obtenir une REAL ID en Californie ?";
  const citations = [
    {
      evidence_id: "E1",
      document_title: "California DMV — Get Your REAL ID",
      section_title: "Documents requis",
      evidence_quote: "Vous devez fournir une preuve d’identité, deux preuves de résidence en Californie et votre numéro de sécurité sociale.",
      span_id: "demo-real-id",
    },
    {
      evidence_id: "E2",
      document_title: "California DMV — REAL ID Checklist",
      section_title: "Liste de contrôle",
      evidence_quote: "Les documents originaux ou certifiés doivent être présentés lors du rendez-vous.",
      span_id: "demo-checklist",
    },
  ];
  const answer = "Pour demander une REAL ID en Californie, préparez une preuve d’identité, deux justificatifs de résidence californienne et votre numéro de sécurité sociale [E1]. Apportez les originaux ou des copies certifiées lors de votre rendez-vous au DMV [E2].";

  state.busy = true;
  elements.demoButton.disabled = true;
  elements.demoButton.textContent = "Démo en cours…";
  updateSendState();

  for (const character of question) {
    if (demoToken !== state.demoToken) return;
    elements.messageInput.value += character;
    autoResize();
    await delay(18);
  }
  await delay(320);
  if (demoToken !== state.demoToken) return;

  appendMessage("user", question);
  elements.messageInput.value = "";
  autoResize();
  state.history.push({ role: "user", content: question });
  startPipeline();
  appendTyping();

  await delay(2600);
  if (demoToken !== state.demoToken) return;

  removeTyping();
  stopPipeline();
  appendMessage("assistant", answer, citations);
  state.history.push({ role: "assistant", content: answer, citations });
  renderSources(citations);
  persistCurrentConversation();
  state.busy = false;
  elements.demoButton.disabled = false;
  elements.demoButton.textContent = "Voir une démo";
  updateSendState();
}

async function submitQuestion(event) {
  event?.preventDefault();
  const message = elements.messageInput.value.trim();
  if (!message || state.busy) return;

  const priorHistory = state.history.map(({ role, content }) => ({ role, content }));
  appendMessage("user", message);
  state.history.push({ role: "user", content: message });
  elements.messageInput.value = "";
  autoResize();
  state.busy = true;
  updateSendState();
  startPipeline();
  appendTyping();

  try {
    const response = await fetch("/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        history: priorHistory.slice(-12),
        domain: state.domain || null,
        use_dense: true,
      }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(data.detail || "Le service n’a pas pu générer de réponse.");
    }

    removeTyping();
    const answer = data.answer || "Je n’ai pas trouvé de réponse suffisamment fiable.";
    const citations = Array.isArray(data.citations) ? data.citations : [];
    const traceId = data.trace_id || "";
    appendMessage("assistant", answer, citations, traceId);
    state.history.push({ role: "assistant", content: answer, citations, trace_id: traceId });
    renderSources(citations);
    persistCurrentConversation();
  } catch (error) {
    removeTyping();
    const messageText = error instanceof Error ? error.message : "Une erreur inattendue est survenue.";
    appendMessage(
      "assistant",
      `Je ne peux pas répondre pour le moment. ${messageText} Vous pouvez vérifier que l’API locale est démarrée, puis réessayer.`,
    );
  } finally {
    stopPipeline();
    state.busy = false;
    updateSendState();
    elements.messageInput.focus();
  }
}

async function sendFeedback(button) {
  const article = button.closest(".message[data-trace-id]");
  const controls = button.closest(".message-feedback");
  const traceId = article?.dataset.traceId || "";
  if (!traceId || !controls || controls.dataset.sent === "true") return;

  const buttons = [...controls.querySelectorAll("button")];
  buttons.forEach((item) => { item.disabled = true; });
  try {
    const response = await fetch("/v1/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trace_id: traceId, rating: button.dataset.feedback }),
    });
    if (!response.ok) throw new Error("feedback rejected");
    controls.dataset.sent = "true";
    controls.innerHTML = "<span>Merci pour votre retour.</span>";
  } catch {
    buttons.forEach((item) => { item.disabled = false; });
    controls.querySelector("span").textContent = "Retour non envoyé. Réessayez.";
  }
}

elements.composer.addEventListener("submit", submitQuestion);
elements.messageInput.addEventListener("input", () => {
  autoResize();
  updateSendState();
});
elements.messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    submitQuestion(event);
  }
});
elements.newChat.addEventListener("click", resetConversation);
elements.historyList.addEventListener("click", (event) => {
  const item = event.target.closest("[data-conversation-id]");
  if (item) restoreConversation(item.dataset.conversationId || "");
});
elements.demoButton.addEventListener("click", runDemo);
elements.openSources.addEventListener("click", () => openSourceDrawer());
elements.closeEvidence.addEventListener("click", closePanels);
elements.mobileOverlay.addEventListener("click", closePanels);
elements.menuButton.addEventListener("click", () => {
  closePanels();
  elements.sidebar.classList.add("open");
  elements.mobileOverlay.classList.add("visible");
});

document.querySelectorAll(".domain-item").forEach((button) => {
  button.addEventListener("click", () => selectDomain(button));
});

document.querySelectorAll(".suggestion-card").forEach((button) => {
  button.addEventListener("click", () => {
    elements.messageInput.value = button.dataset.question || "";
    autoResize();
    updateSendState();
    elements.messageInput.focus();
  });
});

elements.messages.addEventListener("click", (event) => {
  const feedback = event.target.closest("[data-feedback]");
  if (feedback) {
    sendFeedback(feedback);
    return;
  }
  const citation = event.target.closest("[data-citation]");
  if (citation) openSourceDrawer(citation.dataset.citation || "");
});

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    resetConversation();
  }
  if (event.key === "Escape") closePanels();
});

autoResize();
updateSendState();
renderSources([]);
loadSavedConversations();
renderConversationHistory();
