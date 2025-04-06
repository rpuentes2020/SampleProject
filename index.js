import * as webllm from "https://esm.run/@mlc-ai/web-llm";

let initalPromptCompleted;

document.addEventListener("DOMContentLoaded", function() {
  // code...
  initializeWebLLMEngine().then(sendInitialPrompt);
});


/*************** WebLLM logic ***************/
const messages = [
  {
    content: "You are a helpful AI agent helping users.",
    role: "system",
  },
];

const availableModels = webllm.prebuiltAppConfig.model_list.map(
  (m) => m.model_id,
);
// let selectedModel = "Llama-3.1-8B-Instruct-q4f32_1-1k";
// let selectedModel = "SmolLM2-1.7B-Instruct-q4f16_1-MLC";
let selectedModel = "Mistral-7B-Instruct-v0.3-q4f16_1-MLC";
// Callback function for initializing progress
function updateEngineInitProgressCallback(report) {
  console.log("initialize", report.progress);
  document.getElementById("download-status").textContent = report.text;
}

// Create engine instance
const engine = new webllm.MLCEngine();
engine.setInitProgressCallback(updateEngineInitProgressCallback);

async function initializeWebLLMEngine() {
  document.getElementById("download-status").classList.remove("hidden");
  selectedModel = document.getElementById("model-selection").value;
  const config = {
    temperature: 1.0,
    top_p: 1,
  };
  await engine.reload(selectedModel, config);
}

async function streamingGenerating(messages, onUpdate, onFinish, onError) {
  try {
    let curMessage = "";
    let usage;
    const completion = await engine.chat.completions.create({
      stream: true,
      messages,
      stream_options: { include_usage: true },
    });
    for await (const chunk of completion) {
      const curDelta = chunk.choices[0]?.delta.content;
      if (curDelta) {
        curMessage += curDelta;
      }
      if (chunk.usage) {
        usage = chunk.usage;
      }
      onUpdate(curMessage);
    }
    const finalMessage = await engine.getMessage();
    onFinish(finalMessage, usage);
  } catch (err) {
    onError(err);
  }
}

/*************** UI logic ***************/
let initialPrompt = `Respond to this passage with True if you understand it and False if you do not. For all future submissions, read all following passages and extract as many factual questions and their direct answers as possible. Return your output as a JSON array where each item is an object with two fields: "question" and "answer". Keep questions clear and concise. Keep answers short, factual, and based only on the passage (no invented info or speculative language). The final output must be a valid JSON array.
[  
  { "question": "Your first question here?", "answer": "Answer here." },  
  { "question": "Your second question?", "answer": "Second answer." }  
]
`
async function sendInitialPrompt(){ 

  document.getElementById("user-input").value = initialPrompt;
  onMessageSend();
  initalPromptCompleted = true;
  document.getElementById("runOcr").removeAttribute("disabled");
}


function onMessageSend() {
  const input = document.getElementById("user-input").value.trim();

  const message = {
    content: input,
    role: "user",
  };
  if (input.length === 0) {
    return;
  }
  document.getElementById("send").disabled = true;

  messages.push(message);
  appendMessage(message);

  document.getElementById("user-input").value = "";
  document
    .getElementById("user-input")
    .setAttribute("placeholder", "Generating...");

  const aiMessage = {
    content: "typing...",
    role: "assistant",
  };
  appendMessage(aiMessage);

  const onFinishGenerating = (finalMessage, usage) => {
    updateLastMessage(finalMessage);
    document.getElementById("send").disabled = false;
    const usageText =
      `prompt_tokens: ${usage.prompt_tokens}, ` +
      `completion_tokens: ${usage.completion_tokens}, ` +
      `prefill: ${usage.extra.prefill_tokens_per_s.toFixed(4)} tokens/sec, ` +
      `decoding: ${usage.extra.decode_tokens_per_s.toFixed(4)} tokens/sec`;
    document.getElementById("chat-stats").classList.remove("hidden");
    document.getElementById("chat-stats").textContent = usageText;
  };

  streamingGenerating(
    messages,
    updateLastMessage,
    onFinishGenerating,
    console.error,
  );
}

function appendMessage(message) {
  const chatBox = document.getElementById("chat-box");
  const container = document.createElement("div");
  container.classList.add("message-container");
  const newMessage = document.createElement("div");
  newMessage.classList.add("message");
  newMessage.textContent = message.content;

  if (message.role === "user") {
    container.classList.add("user");
  } else {
    container.classList.add("assistant");
  }

  container.appendChild(newMessage);
  chatBox.appendChild(container);
  chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the latest message
}

function updateLastMessage(content) {
  const messageDoms = document
    .getElementById("chat-box")
    .querySelectorAll(".message");
  const lastMessageDom = messageDoms[messageDoms.length - 1];
  lastMessageDom.textContent = content;
}

/*************** UI binding ***************/
availableModels.forEach((modelId) => {
  const option = document.createElement("option");
  option.value = modelId;
  option.textContent = modelId;
  document.getElementById("model-selection").appendChild(option);
});
document.getElementById("model-selection").value = selectedModel;
document.getElementById("download").addEventListener("click", function () {
  initializeWebLLMEngine().then(() => {
    document.getElementById("send").disabled = false;
  });
});
document.getElementById("send").addEventListener("click", function () {
  onMessageSend();
});


// **************************************************************************************************
// OCR LOGIC
// **************************************************************************************************

const fileInput = document.getElementById("fileInput");
const runOcrBtn = document.getElementById("runOcr");
const status = document.getElementById("status");
const output = document.getElementById("ocrText");
const canvas = document.getElementById("pdfCanvas");
const ctx = canvas.getContext("2d");


fileInput.addEventListener("change", () => {
  if (fileInput.files.length > 0) {
    status.textContent = "📂 File ready. Click 'Extract Text'.";
    runOcrBtn.disabled = false;
  } else {
    status.textContent = "Waiting for file...";
    runOcrBtn.disabled = true;
  }
});


runOcrBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  const status = document.getElementById("status");
  const output = document.getElementById("ocrText");
  const AiMessageInput = document.getElementById("user-input");

  if (!file) return alert("Please choose a file.");

  status.textContent = "🧠 Extracting text...";
  output.value = "";

  const type = file.type;
  let cleanedText;


  if (type === "application/pdf") {
    const text = await extractFromPDF(file);
    cleanedText = cleanOcrText(text);
    AiMessageInput.value = cleanedText;
    output.value = cleanedText;
    onMessageSend();
  } else if (type.startsWith("image/")) {
    const result = await Tesseract.recognize(file, "eng");
    cleanedText = cleanOcrText(result.data.text);
    output.value = cleanedText;
    AiMessageInput.value = cleanedText;
    onMessageSend();
  } else {
    status.textContent = "❌ Unsupported file type.";
    return;
  }

  status.textContent = "✅ Extraction complete.";
});



// Attempt to extract readable text from PDF, fallback to OCR if needed
async function extractFromPDF(file) {
  const buffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: buffer }).promise;
  let finalText = "";

  for (let pageNum = 1; pageNum <= Math.min(3, pdf.numPages); pageNum++) {
    const page = await pdf.getPage(pageNum);
    const content = await page.getTextContent();
    const extracted = content.items.map(item => item.str).join(" ").trim();

    if (extracted.length > 30) {
      finalText += extracted + "\n\n";
    } else {
      const viewport = page.getViewport({ scale: 2 });
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      await page.render({ canvasContext: ctx, viewport }).promise;
      const imgData = canvas.toDataURL("image/png");
      const { data: { text } } = await Tesseract.recognize(imgData, "eng");
      finalText += text + "\n\n";
    }
  }

  return finalText;
}

// Optional: clean up common OCR artifacts
// function cleanOcrText(text) {
//   return text
//     .replace(/Figure\s+\d+(\.\d+)?/gi, "")
//     .replace(/^[^a-zA-Z0-9]{2,}.*$/gim, "")
//     .replace(/^[A-Z\s\W]{5,}$/gm, "")
//     .replace(/^\s*\w{1,10}\s*$/gm, "")
//     .replace(/[”“„‟″‶‷]/g, '"')
//     .replace(/[’‘‚‛]/g, "'")
//     .replace(/(\w)-\n(\w)/g, "$1$2")
//     .replace(/(?<!\n)\n(?!\n)/g, " ")
//     .replace(/\n{2,}/g, "\n\n")
//     .replace(/\s{2,}/g, " ")
//     .replace(/^\s+/gm, "")
//     .replace(/\s+$/gm, "")
//     .replace(/^\d+\s*\|\s*Chapter\s+\d+/im, "")
//     .trim();
// }

// old ocr logic


// document.getElementById("runOcr").addEventListener("click", async () => {
//   const file = document.getElementById("imageUpload").files[0];
//   const status = document.getElementById("status");
//   const output = document.getElementById("ocrText");
//   const AiMessageInput = document.getElementById("user-input");

//   if (!file) {
//     alert("Please upload an image first.");
//     return;
//   }

//   status.textContent = "🔄 Running OCR... please wait.";
//   output.value = "";

//   try {
//     const result = await Tesseract.recognize(file, 'eng');
//     const cleaned = cleanOcrText(result.data.text);
//     output.value = cleaned;
//     // output.value = result.data.text.trim();
    
//     AiMessageInput.value = cleaned;
//     // AiMessageInput.value = result.data.text.trim();

//     onMessageSend();
//     status.textContent = "✅ OCR completed.";
//   } catch (err) {
//     console.error(err);
//     status.textContent = "❌ OCR failed. See console for details.";
//   }
// });


function cleanOcrText(text) {
  return text
    .replace(/[^A-Za-z0-9.,;:'"()\-–—\s\n]/g, "")        // Remove weird non-word symbols
    .replace(/(\n\s*\n)+/g, "\n\n")                      // Collapse extra line breaks
    .replace(/(?<=\w)-\s*\n\s*/g, "")                    // Join hyphenated line breaks
    .replace(/[|=_]{2,}/g, "")                           // Remove divider-like characters
    .replace(/^\s*[a-zA-Z]?\s*$/, "")                    // Remove lone letters/initials per line
    .replace(/^\s*Figure\s+\d+.*$/gmi, "")               // Remove figure labels
    .replace(/\s{2,}/g, " ")                             // Collapse extra spaces
    .replace(/\n +/g, "\n")                              // Remove leading spaces per line
     // 1. Remove figure captions
     .replace(/Figure\s+\d+(\.\d+)?/gi, "")
     // 2. Remove lines that are mostly symbols or capital gibberish
     .replace(/^[^a-zA-Z0-9]{2,}.*$/gim, "")
     .replace(/^[A-Z\s\W]{5,}$/gm, "")
     // 3. Remove lone lines with 1–2 words (e.g., "NN", "Cilla")
     .replace(/^\s*\w{1,10}\s*$/gm, "")
     // 4. Remove OCR misreads like double punctuation
     .replace(/[”“„‟″‶‷]/g, '"')
     .replace(/[’‘‚‛]/g, "'")
     // 5. Join hyphenated line breaks
     .replace(/(\w)-\n(\w)/g, "$1$2")
     // 6. Fix double newlines or newlines in the middle of sentences
     .replace(/(?<!\n)\n(?!\n)/g, " ") // Replace single newlines with space
     .replace(/\n{2,}/g, "\n\n")       // Keep paragraph breaks
     // 7. Remove extra spaces
     .replace(/\s{2,}/g, " ")
     .replace(/^\s+/gm, "")            // Trim line starts
     .replace(/\s+$/gm, "")            // Trim line ends
     // 8. Remove page headers like "86 | Chapter 5"
     .replace(/^\d+\s*\|\s*Chapter\s+\d+/im, "")
    .trim();
}


