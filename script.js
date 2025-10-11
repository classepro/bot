// Fonction pour charger dynamiquement le HTML
function loadHTML() {
    const appContainer = document.getElementById('appContainer');
    
    const htmlContent = `
      <div class="chat-container">
        <div class="chat-header">
          <h1>Professeur ClassePro <span class="professor-badge">Assistant Pédagogique</span></h1>
          <div class="task-selector" id="taskSelector">
            <button class="task-btn" data-type="cours">📚 Cours</button>
            <button class="task-btn" data-type="explication">💡 Explication</button>
            <button class="task-btn" data-type="exercices">✏️ Exercices</button>
            <button class="task-btn" data-type="resume">📋 Résumé</button>
            <button class="task-btn" data-type="qcm">❓ QCM</button>
            <button class="task-btn" data-type="dissertation">📝 Dissertation</button>
            <button class="task-btn" data-type="correction">✍️ Correction</button>
            <button class="task-btn" data-type="analyse">🖼️ Analyse</button>
            <button class="task-btn" data-type="maths">➗ Maths</button>
            <button class="task-btn" data-type="fiche">📄 Fiche</button>
          </div>
        </div>
        
        <div class="chat-messages" id="chatMessages">
          <!-- Messages dynamiques -->
          <div class="message bot">
            <div class="bot-message-content">
              <strong>Bonjour ! 👋</strong> Je suis votre <span class="professor-badge">Professeur ClassePro</span>. Je suis là pour vous aider à réviser vos cours, comprendre vos leçons et progresser dans vos apprentissages. 
              <br><br>
              <strong>Sélectionnez le type de contenu que vous souhaitez générer :</strong>
              <br>
              • <strong>Cours complet</strong> - Structure détaillée avec exemples
              <br>
              • <strong>Explication simplifiée</strong> - Pour comprendre facilement
              <br>
              • <strong>Exercices avec corrigés</strong> - Pour s'entraîner
              <br>
              • <strong>Résumé</strong> - Pour réviser rapidement
              <br>
              • <strong>QCM interactif</strong> - Pour tester ses connaissances
              <br>
              • <strong>Dissertation/Exposé</strong> - Structure académique complète
              <br>
              • <strong>Correction de texte</strong> - Amélioration orthographe/grammaire
              <br>
              • <strong>Analyse d'image</strong> - Description et interprétation
              <br>
              • <strong>Exercices de maths</strong> - Avec solutions détaillées
              <br>
              • <strong>Fiche de révision</strong> - Points clés essentiels
              <br><br>
              Quelle matière ou quel sujet souhaitez-vous travailler aujourd'hui ?
            </div>
          </div>
        </div>
      </div>

      <div class="chat-input-container">
        <div class="chat-input-wrapper">
          <div class="chat-input">
            <textarea id="messageInput" placeholder="Indiquez le sujet, le texte ou l'URL..." rows="1"></textarea>
            <div class="chat-input-actions">
              <input type="file" id="fileInput" accept="image/*,audio/*,.pdf,.doc,.docx,.xls,.xlsx,.ppt,.pptx,.txt" hidden multiple>
              <button type="button" id="uploadBtn" title="Envoyer un fichier">📎</button>
              <button type="button" id="sendBtn">➤</button>
            </div>
          </div>
        </div>
      </div>

      <!-- Modal pour la prévisualisation des images -->
      <div class="modal" id="imageModal">
        <span class="modal-close" id="modalClose">&times;</span>
        <div class="modal-content" id="modalContent"></div>
      </div>
    `;

    appContainer.innerHTML = htmlContent;
    appContainer.style.display = 'block';
}

// Configuration Groq (à remplacer par votre clé API)
const GROQ_API_KEY = "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";
const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";
const MODEL = "meta-llama/llama-4-scout-17b-16e-instruct";

// Variables pour gérer les limites d'API
let apiErrorCount = 0;
let lastApiCall = 0;
const API_COOLDOWN = 2000; // 2 secondes entre les appels

// Type de tâche sélectionné
let currentTaskType = "cours";

// Prompts optimisés pour chaque type de tâche
const TASK_PROMPTS = {
  cours: `Rédige un cours complet sur le sujet : "{sujet}".
Structure : Introduction, I. Grand titre, 1. Petit titre, Texte, ..., Conclusion.
Chaque grand titre doit avoir exactement trois petits titres avec arguments détaillés et exemples.
Langage clair et pédagogique, adapté pour un lycéen.
Utilise des balises HTML sémantiques sans symboles Markdown.`,

  explication: `Explique le sujet : "{sujet}" pour un élève de lycée.
Utilise des phrases simples et courtes.
Illustre chaque idée par un exemple concret.
Structure en sections claires.
Utilise des balises HTML sémantiques sans symboles Markdown.`,

  exercices: `Crée 5 exercices sur : "{sujet}".
Chaque exercice doit avoir une réponse détaillée avec explication de la méthode.
Format structuré avec énoncés clairs et solutions complètes.
Utilise des balises HTML sémantiques sans symboles Markdown.`,

  resume: `Résume le texte ou le sujet : "{sujet}".
Structure : Introduction, Points clés, Conclusion.
Texte clair, concis, adapté pour révision rapide.
Utilise des balises HTML sémantiques sans symboles Markdown.`,

  qcm: `Crée un QCM de 5 questions sur : "{sujet}".
4 options (A, B, C, D) par question.
Indique la bonne réponse et explique pourquoi.
Format interactif et pédagogique.
Utilise des balises HTML sémantiques sans symboles Markdown.`,

  dissertation: `Rédige une dissertation sur : "{sujet}".
Introduction : enjeux et problématique.
Développement : I, II, III grands titres avec 3 sous-titres chacun (arguments + exemples).
Conclusion : résumé et ouverture.
Structure académique complète.
Utilise des balises HTML sémantiques sans symboles Markdown.`,

  correction: `Corrige le texte : "{sujet}".
Orthographe, grammaire, ponctuation.
Explique les corrections et suggère des améliorations.
Sois bienveillant et pédagogique.
Utilise des balises HTML sémantiques sans symboles Markdown.`,

  analyse: `Analyse l'image ou le contenu : "{sujet}".
Décris ce qu'il représente.
Explique le contexte ou le concept.
Donne un exemple concret si possible.
Approche pédagogique et accessible.
Utilise des balises HTML sémantiques sans symboles Markdown.`,

  maths: `Crée 5 exercices de mathématiques sur : "{sujet}".
Chaque exercice avec énoncé et solution détaillée.
Fournis les étapes de raisonnement clairement.
Adapté au niveau lycée.
Utilise des balises HTML sémantiques sans symboles Markdown.`,

  fiche: `Crée une fiche de révision sur : "{sujet}".
Points clés résumés en 5 sections principales.
Chaque point clé expliqué en une phrase et un exemple.
Format condensé et efficace.
Utilise des balises HTML sémantiques sans symboles Markdown.`
};

// Éléments DOM
let chatMessages, messageInput, uploadBtn, fileInput, sendBtn, imageModal, modalContent, modalClose, taskSelector;

// Historique des messages avec prompt système intégré
let messageHistory = [
  {
    role: "system",
    content: `Tu es "Professeur ClassePro", un assistant éducatif qui aide les élèves à réviser leurs cours et à comprendre leurs leçons. Tu expliques avec des mots simples et tu proposes des exercices interactifs.

TON IDENTITÉ:
Tu es "Professeur ClassePro", un enseignant patient, clair et structuré.
Tu as été conçu par l'équipe ClassePro pour aider les élèves.
Tu es spécialisé dans l'explication pédagogique et l'accompagnement scolaire.

MÉTHODE PÉDAGOGIQUE:
Tu présentes toujours les cours avec une organisation logique : introduction, développement structuré, exemples concrets, conclusion claire.
Tu n'utilises JAMAIS les symboles **, ###, ##, #, --, ou tout autre signe de mise en forme Markdown.
Tu utilises le gras HTML <strong> pour mettre en valeur les termes importants.
Tu écris toujours dans un français académique adapté aux élèves du collège ou du lycée.
Tu expliques avec des mots simples, puis tu approfondis avec des exemples.

RÈGLES STRICTES DE MISE EN FORME:
Tu ne dois JAMAIS utiliser **texte** pour le gras.
Tu ne dois JAMAIS utiliser ###, ##, # pour les titres.
Tu ne dois JAMAIS utiliser -- ou --- pour les séparateurs.
Tu utilises <strong>texte</strong> pour mettre en valeur les termes importants.
Tu structures ton texte avec des titres clairs sans symboles.
Tu utilises des paragraphes aérés et bien organisés.
Tu adaptes ton langage au niveau de l'élève.
Tu es patient et encourageant dans tes explications.
Tu utilises des exemples concrets et des analogies simples.

Maintenant, accueille l'élève comme le Professeur ClassePro.`
  },
  {
    role: "assistant",
    content: "Bonjour ! 👋 Je suis votre Professeur ClassePro. Je suis là pour vous aider à réviser vos cours, comprendre vos leçons et progresser dans vos apprentissages. \n\nSélectionnez le type de contenu que vous souhaitez générer :\n• Cours complet - Structure détaillée avec exemples\n• Explication simplifiée - Pour comprendre facilement\n• Exercices avec corrigés - Pour s'entraîner\n• Résumé - Pour réviser rapidement\n• QCM interactif - Pour tester ses connaissances\n• Dissertation/Exposé - Structure académique complète\n• Correction de texte - Amélioration orthographe/grammaire\n• Analyse d'image - Description et interprétation\n• Exercices de maths - Avec solutions détaillées\n• Fiche de révision - Points clés essentiels\n\nQuelle matière ou quel sujet souhaitez-vous travailler aujourd'hui ?"
  }
];

// Fonction pour initialiser les éléments DOM après le chargement du HTML
function initializeDOMElements() {
  chatMessages = document.getElementById("chatMessages");
  messageInput = document.getElementById("messageInput");
  uploadBtn = document.getElementById("uploadBtn");
  fileInput = document.getElementById("fileInput");
  sendBtn = document.getElementById("sendBtn");
  imageModal = document.getElementById("imageModal");
  modalContent = document.getElementById("modalContent");
  modalClose = document.getElementById("modalClose");
  taskSelector = document.getElementById("taskSelector");
}

// Initialiser le scroll vers le bas
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Gestion de la sélection du type de tâche
function setupTaskSelector() {
  taskSelector.addEventListener("click", (e) => {
    if (e.target.classList.contains("task-btn")) {
      // Retirer la classe active de tous les boutons
      document.querySelectorAll(".task-btn").forEach(btn => {
        btn.classList.remove("active");
      });
      
      // Ajouter la classe active au bouton cliqué
      e.target.classList.add("active");
      
      // Mettre à jour le type de tâche courant
      currentTaskType = e.target.dataset.type;
      
      // Mettre à jour le placeholder
      updatePlaceholder();
    }
  });
}

// Fonction pour mettre à jour le placeholder en fonction du type de tâche
function updatePlaceholder() {
  const placeholders = {
    cours: "Entrez le sujet du cours (ex: Révolution française)...",
    explication: "Entrez le concept à expliquer (ex: Photosynthèse)...",
    exercices: "Entrez le thème des exercices (ex: Equations du second degré)...",
    resume: "Collez le texte à résumer ou le sujet...",
    qcm: "Entrez le thème du QCM (ex: Grammaire française)...",
    dissertation: "Entrez le sujet de dissertation (ex: La liberté)...",
    correction: "Collez le texte à corriger...",
    analyse: "Décrivez l'image ou collez l'URL...",
    maths: "Entrez le chapitre mathématique (ex: Trigonométrie)...",
    fiche: "Entrez le sujet de la fiche (ex: Cellule vivante)..."
  };
  
  messageInput.placeholder = placeholders[currentTaskType] || "Indiquez le sujet, le texte ou l'URL...";
}

// Gestion de l'envoi du message
function setupMessageHandlers() {
  sendBtn.addEventListener("click", sendMessage);
  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

// Gestion du bouton d'upload
function setupFileUpload() {
  uploadBtn.addEventListener("click", () => fileInput.click());

  // Gestion de la sélection de fichiers
  fileInput.addEventListener("change", () => {
    const files = fileInput.files;
    if (files.length > 0) {
      for (let i = 0; i < files.length; i++) {
        displayFile(files[i]);
      }
      fileInput.value = "";
    }
  });
}

// Gestion de la fermeture du modal
function setupModalHandlers() {
  modalClose.addEventListener("click", () => {
    imageModal.style.display = "none";
  });

  // Fermer le modal en cliquant à l'extérieur
  imageModal.addEventListener("click", (e) => {
    if (e.target === imageModal) {
      imageModal.style.display = "none";
    }
  });
}

// Fonction pour envoyer un message
async function sendMessage() {
  const userMessage = messageInput.value.trim();
  if (userMessage) {
    // Formater le message avec le prompt de la tâche sélectionnée
    const taskPrompt = TASK_PROMPTS[currentTaskType].replace("{sujet}", userMessage);
    const finalMessage = `[Tâche: ${currentTaskType}] ${taskPrompt}`;
    
    addMessage(userMessage, "user");
    messageInput.value = "";
    resetTextareaHeight();

    // Ajouter le message formaté à l'historique
    messageHistory.push({ role: "user", content: finalMessage });

    // Vérifier le cooldown de l'API
    const now = Date.now();
    if (now - lastApiCall < API_COOLDOWN) {
      const waitTime = API_COOLDOWN - (now - lastApiCall);
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }

    // Afficher l'indicateur de frappe
    const typingIndicator = addTypingIndicator();

    try {
      // Appeler l'API Groq
      lastApiCall = Date.now();
      const response = await callGroqAPI(finalMessage);
      
      // Supprimer l'indicateur de frappe
      if (typingIndicator && typingIndicator.parentNode) {
        typingIndicator.parentNode.remove();
      }
      
      if (response) {
        // Ajouter la réponse à l'historique et l'afficher
        messageHistory.push({ role: "assistant", content: response });
        addMessage(response, "bot");
        apiErrorCount = 0; // Réinitialiser le compteur d'erreurs
      } else {
        throw new Error("Aucune réponse de l'API Groq");
      }
    } catch (error) {
      console.error("Erreur:", error);
      apiErrorCount++;
      
      // Supprimer l'indicateur de frappe en cas d'erreur
      if (typingIndicator && typingIndicator.parentNode) {
        typingIndicator.parentNode.remove();
      }
      
      handleGenericError(userMessage);
    }
  }
}

// Fonction pour appeler l'API Groq
async function callGroqAPI(userPrompt) {
  try {
    const response = await fetch(GROQ_API_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${GROQ_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        model: MODEL,
        messages: messageHistory,
        temperature: 0.7,
        max_tokens: 4096
      })
    });

    if (!response.ok) {
      throw new Error(`Erreur API Groq: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    if (data.error) {
      console.error("Erreur API Groq :", data.error.message);
      return null;
    }

    return data.choices[0].message.content.trim();
    
  } catch (error) {
    console.error("Erreur lors de l'appel à Groq :", error);
    throw error;
  }
}

// Fonction pour gérer les erreurs génériques
function handleGenericError(originalMessage) {
  const errorMsg = document.createElement("div");
  errorMsg.classList.add("message", "bot");
  errorMsg.innerHTML = `
    <div class="bot-message-content">
      <div class="error-message">
        <strong>Problème de connexion</strong><br>
        Impossible de contacter le service pour le moment. 
        Voici une réponse générée localement :<br><br>
        <strong>En tant que Professeur ClassePro</strong>, je rencontre des difficultés techniques. 
        ${generateFallbackResponse(originalMessage)}
        <br><br>
        <button class="retry-btn" onclick="retryLastMessage()">Réessayer</button>
      </div>
    </div>
  `;
  chatMessages.appendChild(errorMsg);
  scrollToBottom();
}

// Fonction pour regénérer une réponse de secours
function generateFallbackResponse(message) {
  const taskTitles = {
    cours: "cours complet",
    explication: "explication simplifiée",
    exercices: "exercices avec corrigés",
    resume: "résumé",
    qcm: "QCM interactif",
    dissertation: "dissertation",
    correction: "correction de texte",
    analyse: "analyse",
    maths: "exercices de maths",
    fiche: "fiche de révision"
  };
  
  return `Je prépare ${taskTitles[currentTaskType]} sur "${message}". Une fois le service rétabli, je pourrai vous fournir un contenu structuré et pédagogique. En attendant, vous pouvez reformuler votre demande ou essayer un autre type de contenu.`;
}

// Fonction pour réessayer le dernier message
window.retryLastMessage = function() {
  // Supprimer le message d'erreur
  const lastMessage = chatMessages.lastChild;
  if (lastMessage && lastMessage.querySelector('.error-message')) {
    lastMessage.remove();
  }
  
  // Réessayer d'envoyer le dernier message utilisateur
  const lastUserMessage = messageHistory[messageHistory.length - 1];
  if (lastUserMessage && lastUserMessage.role === 'user') {
    sendMessageToAPI(lastUserMessage.content);
  }
};

// Fonction pour envoyer un message à l'API (séparée pour la réutilisation)
async function sendMessageToAPI(message) {
  const typingIndicator = addTypingIndicator();
  
  try {
    lastApiCall = Date.now();
    const response = await callGroqAPI(message);
    
    if (typingIndicator && typingIndicator.parentNode) {
      typingIndicator.parentNode.remove();
    }
    
    if (response) {
      messageHistory.push({ role: "assistant", content: response });
      addMessage(response, "bot");
      apiErrorCount = 0;
    } else {
      throw new Error("Aucune réponse de l'API Groq");
    }
  } catch (error) {
    console.error("Erreur lors de la nouvelle tentative:", error);
    if (typingIndicator && typingIndicator.parentNode) {
      typingIndicator.parentNode.remove();
    }
    handleGenericError(message);
  }
}

// Fonction pour formater le texte avec une structure académique
function formatAcademicText(text) {
  // Nettoyer le texte de tous les symboles Markdown
  let cleanedText = text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')  // **gras** -> <strong>gras</strong>
    .replace(/\*(.*?)\*/g, '<strong>$1</strong>')       // *gras* -> <strong>gras</strong>
    .replace(/### (.*?)(\n|$)/g, '<div class="section-title">$1</div>')  // ### Titre
    .replace(/## (.*?)(\n|$)/g, '<div class="section-title">$1</div>')   // ## Titre
    .replace(/# (.*?)(\n|$)/g, '<div class="section-title">$1</div>')    // # Titre
    .replace(/---+/g, '')  // Supprimer les lignes de séparation
    .replace(/--/g, '')    // Supprimer les tirets doubles
    .replace(/\[(.*?)\]\((.*?)\)/g, '$1');  // [lien](url) -> lien
  
  // Séparer le texte en paragraphes
  const paragraphs = cleanedText.split('\n\n');
  
  let formattedHTML = '';
  
  paragraphs.forEach(paragraph => {
    const trimmed = paragraph.trim();
    if (trimmed.length === 0) return;
    
    // Détecter les exercices et solutions
    if (trimmed.toLowerCase().includes('exercice') || trimmed.match(/exercice\s*\d+/i)) {
      formattedHTML += `<div class="exercise">${trimmed}</div>`;
    }
    else if (trimmed.toLowerCase().includes('solution') || trimmed.toLowerCase().includes('corrigé')) {
      formattedHTML += `<div class="solution">${trimmed}</div>`;
    }
    else if (trimmed.match(/^[abcd]\)/i) || trimmed.match(/^option\s*[abcd]/i)) {
      formattedHTML += `<div class="qcm-option">${trimmed}</div>`;
    }
    else if (trimmed.toLowerCase().includes('bonne réponse') || trimmed.toLowerCase().includes('correct')) {
      formattedHTML += `<div class="correct-answer">${trimmed}</div>`;
    }
    // Détecter les titres de sections naturels (sans symboles)
    else if (trimmed.match(/^(INTRODUCTION|Introduction|INTRODUCTION:|Introduction:|CONCLUSION|Conclusion|CONCLUSION:|Conclusion:|I\.|II\.|III\.|IV\.|V\.|1\.|2\.|3\.|4\.|5\.|•|Exemple|EXEMPLE)/i)) {
      if (trimmed.toLowerCase().includes('introduction')) {
        formattedHTML += `<div class="introduction">${trimmed}</div>`;
      } else if (trimmed.toLowerCase().includes('conclusion')) {
        formattedHTML += `<div class="conclusion">${trimmed}</div>`;
      } else if (trimmed.match(/^[IVX]+\./)) {
        formattedHTML += `<div class="main-point"><div class="main-point-title">${trimmed}</div></div>`;
      } else if (trimmed.match(/^[0-9]+\./)) {
        formattedHTML += `<div class="sub-point"><span class="sub-point-title">${trimmed}</span></div>`;
      } else if (trimmed.startsWith('•')) {
        formattedHTML += `<div class="sub-point">${trimmed}</div>`;
      } else if (trimmed.toLowerCase().includes('exemple')) {
        formattedHTML += `<div class="example">${trimmed}</div>`;
      } else {
        formattedHTML += `<p>${trimmed}</p>`;
      }
    } 
    // Détecter les exemples
    else if (trimmed.toLowerCase().includes('exemple') || trimmed.toLowerCase().includes('par exemple')) {
      formattedHTML += `<div class="example">${trimmed}</div>`;
    }
    // Détecter les titres naturels (sans symboles Markdown)
    else if (trimmed.endsWith(':') && trimmed.length < 50 && !trimmed.includes(' ')) {
      formattedHTML += `<div class="section-title">${trimmed}</div>`;
    }
    else {
      formattedHTML += `<p>${trimmed}</p>`;
    }
  });
  
  return formattedHTML;
}

// Fonction pour ajouter un indicateur de frappe
function addTypingIndicator() {
  const msg = document.createElement("div");
  msg.classList.add("message", "bot");
  
  const content = document.createElement("div");
  content.classList.add("bot-message-content");
  
  const typingIndicator = document.createElement("span");
  typingIndicator.classList.add("typing-indicator");
  typingIndicator.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
  
  content.appendChild(typingIndicator);
  msg.appendChild(content);
  chatMessages.appendChild(msg);
  scrollToBottom();
  
  return typingIndicator;
}

// Fonction pour ajuster automatiquement la hauteur du textarea
function setupTextareaAutoResize() {
  messageInput.addEventListener("input", function() {
    this.style.height = "auto";
    this.style.height = (this.scrollHeight) + "px";
  });
}

// Fonction pour réinitialiser la hauteur du textarea
function resetTextareaHeight() {
  messageInput.style.height = "auto";
}

// Fonction pour afficher un fichier dans la conversation
function displayFile(file) {
  const preview = document.createElement("div");
  preview.classList.add("message", "user", "file-preview");

  // Obtenir l'icône appropriée selon le type de fichier
  const fileIcon = getFileIcon(file.type);
  
  if (file.type.startsWith("image/")) {
    // Convertir l'image en base64
    const reader = new FileReader();
    reader.onload = function(e) {
      const imgContainer = document.createElement("div");
      imgContainer.classList.add("file-preview-container");
      
      const img = document.createElement("img");
      img.src = e.target.result;
      img.style.cursor = "pointer";
      img.addEventListener("click", () => openImageModal(e.target.result));
      imgContainer.appendChild(img);
      
      const fileInfo = createFileInfo(file, fileIcon);
      imgContainer.appendChild(fileInfo);
      
      preview.appendChild(imgContainer);
      chatMessages.appendChild(preview);
      scrollToBottom();
    };
    reader.readAsDataURL(file);
  } else if (file.type.startsWith("audio/")) {
    const reader = new FileReader();
    reader.onload = function(e) {
      const audio = document.createElement("audio");
      audio.controls = true;
      audio.src = e.target.result;
      preview.appendChild(audio);
      
      const fileInfo = createFileInfo(file, fileIcon);
      preview.appendChild(fileInfo);
      
      chatMessages.appendChild(preview);
      scrollToBottom();
    };
    reader.readAsDataURL(file);
  } else if (file.type === "application/pdf") {
    const reader = new FileReader();
    reader.onload = function(e) {
      const pdfContainer = document.createElement("div");
      pdfContainer.classList.add("file-preview-container");
      
      const pdfIcon = document.createElement("div");
      pdfIcon.innerHTML = "📕";
      pdfIcon.style.fontSize = "48px";
      pdfIcon.style.textAlign = "center";
      pdfIcon.style.marginBottom = "10px";
      pdfContainer.appendChild(pdfIcon);
      
      const fileInfo = createFileInfo(file, fileIcon);
      fileInfo.style.textAlign = "center";
      pdfContainer.appendChild(fileInfo);
      
      // Ajouter un bouton pour visualiser le PDF
      const viewBtn = document.createElement("button");
      viewBtn.textContent = "Ouvrir le PDF";
      viewBtn.style.marginTop = "10px";
      viewBtn.style.padding = "5px 10px";
      viewBtn.style.border = "none";
      viewBtn.style.borderRadius = "5px";
      viewBtn.style.background = "linear-gradient(135deg, #3D3B8E, #FF7E5F)";
      viewBtn.style.color = "white";
      viewBtn.style.cursor = "pointer";
      viewBtn.addEventListener("click", () => window.open(e.target.result, "_blank"));
      pdfContainer.appendChild(viewBtn);
      
      preview.appendChild(pdfContainer);
      chatMessages.appendChild(preview);
      scrollToBottom();
    };
    reader.readAsDataURL(file);
  } else {
    // Pour les autres types de fichiers (Word, Excel, etc.)
    const reader = new FileReader();
    reader.onload = function(e) {
      const fileContainer = document.createElement("div");
      fileContainer.classList.add("file-preview-container");
      
      const fileIconElement = document.createElement("div");
      fileIconElement.innerHTML = fileIcon;
      fileIconElement.style.fontSize = "48px";
      fileIconElement.style.textAlign = "center";
      fileIconElement.style.marginBottom = "10px";
      fileContainer.appendChild(fileIconElement);
      
      const fileInfo = createFileInfo(file, fileIcon);
      fileInfo.style.textAlign = "center";
      fileContainer.appendChild(fileInfo);
      
      // Ajouter un bouton pour télécharger le fichier
      const downloadBtn = document.createElement("button");
      downloadBtn.textContent = "Télécharger";
      downloadBtn.style.marginTop = "10px";
      downloadBtn.style.padding = "5px 10px";
      downloadBtn.style.border = "none";
      downloadBtn.style.borderRadius = "5px";
      downloadBtn.style.background = "linear-gradient(135deg, #3D3B8E, #FF7E5F)";
      downloadBtn.style.color = "white";
      downloadBtn.style.cursor = "pointer";
      downloadBtn.addEventListener("click", () => {
        const a = document.createElement("a");
        a.href = e.target.result;
        a.download = file.name;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      });
      fileContainer.appendChild(downloadBtn);
      
      preview.appendChild(fileContainer);
      chatMessages.appendChild(preview);
      scrollToBottom();
    };
    reader.readAsDataURL(file);
  }
}

// Fonction pour créer l'information du fichier
function createFileInfo(file, icon) {
  const fileInfo = document.createElement("div");
  fileInfo.classList.add("file-item");
  
  const iconSpan = document.createElement("span");
  iconSpan.classList.add("file-icon");
  iconSpan.textContent = icon;
  
  const nameSpan = document.createElement("span");
  nameSpan.classList.add("file-name");
  nameSpan.textContent = file.name;
  
  const sizeSpan = document.createElement("span");
  sizeSpan.classList.add("file-size");
  sizeSpan.textContent = formatFileSize(file.size);
  
  fileInfo.appendChild(iconSpan);
  fileInfo.appendChild(nameSpan);
  fileInfo.appendChild(sizeSpan);
  
  return fileInfo;
}

// Fonction pour obtenir l'icône appropriée selon le type de fichier
function getFileIcon(fileType) {
  if (fileType.startsWith("image/")) return "🖼️";
  if (fileType.startsWith("audio/")) return "🎵";
  if (fileType === "application/pdf") return "📕";
  if (fileType.includes("word") || fileType === "application/msword" || fileType === "application/vnd.openxmlformats-officedocument.wordprocessingml.document") return "📄";
  if (fileType.includes("excel") || fileType === "application/vnd.ms-excel" || fileType === "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet") return "📊";
  if (fileType.includes("powerpoint") || fileType === "application/vnd.ms-powerpoint" || fileType === "application/vnd.openxmlformats-officedocument.presentationml.presentation") return "📑";
  if (fileType.includes("text/plain")) return "📝";
  return "📎";
}

// Fonction pour formater la taille du fichier
function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

// Fonction pour ouvrir le modal d'image
function openImageModal(imageUrl) {
  modalContent.innerHTML = "";
  const img = document.createElement("img");
  img.src = imageUrl;
  modalContent.appendChild(img);
  imageModal.style.display = "flex";
}

// Fonction pour ajouter un message texte
function addMessage(text, sender) {
  const msg = document.createElement("div");
  msg.classList.add("message", sender);
  
  if (sender === "bot") {
    const content = document.createElement("div");
    content.classList.add("bot-message-content");
    // Utiliser innerHTML pour permettre le formatage académique
    content.innerHTML = formatAcademicText(text);
    msg.appendChild(content);
  } else {
    msg.textContent = text;
  }
  
  chatMessages.appendChild(msg);
  scrollToBottom();
}

// Fonction pour initialiser l'application
function initApp() {
  // Masquer l'écran de chargement
  document.getElementById('loadingContainer').style.display = 'none';
  
  // Charger le HTML
  loadHTML();
  
  // Initialiser les éléments DOM
  initializeDOMElements();
  
  // Configurer les gestionnaires d'événements
  setupTaskSelector();
  setupMessageHandlers();
  setupFileUpload();
  setupModalHandlers();
  setupTextareaAutoResize();
  
  // Initialiser le placeholder
  updatePlaceholder();
  
  // Sélectionner le premier bouton par défaut
  document.querySelector('.task-btn').classList.add('active');
  
  console.log("Professeur ClassePro initialisé avec API Groq et LLaMA 4 Scout");
}

// Démarrer l'application après un délai pour simuler le chargement
setTimeout(initApp, 1500);