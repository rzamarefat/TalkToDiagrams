"use client";

import { useState, useRef, useEffect, useCallback } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

type Tab = {
  id: string;
  label: string;
  messages: Message[];
  images: string[];      // server paths → sent to backend
  imageUrls: string[];   // http URLs   → shown in carousel
  carouselIndex: number;
};

function createTab(index: number): Tab {
  return {
    id: crypto.randomUUID(),
    label: `Chat ${index}`,
    messages: [],
    images: [],
    imageUrls: [],
    carouselIndex: 0,
  };
}

export default function Home() {
  const [tabs, setTabs] = useState<Tab[]>([createTab(1)]);
  const [activeTabId, setActiveTabId] = useState<string>(tabs[0].id);
  const [question, setQuestion] = useState("");
  const [imageInput, setImageInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [recording, setRecording] = useState(false);
  const [, forceUpdate] = useState(0); // used to re-render when pendingAudioRef changes
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const pendingAudioRef = useRef<Blob | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const activeTab = tabs.find((t) => t.id === activeTabId)!;

  const updateTab = useCallback(
    (id: string, updater: (tab: Tab) => Tab) => {
      setTabs((prev) => prev.map((t) => (t.id === id ? updater(t) : t)));
    },
    []
  );

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [activeTab?.messages]);

  const addTab = () => {
    const newTab = createTab(tabs.length + 1);
    setTabs((prev) => [...prev, newTab]);
    setActiveTabId(newTab.id);
    setQuestion("");
  };

  const removeTab = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (tabs.length === 1) return;
    const idx = tabs.findIndex((t) => t.id === id);
    const newTabs = tabs.filter((t) => t.id !== id);
    setTabs(newTabs);
    if (activeTabId === id) {
      setActiveTabId(newTabs[Math.max(0, idx - 1)].id);
    }
  };

  const streamResponse = async (
    tabId: string,
    fetchCall: () => Promise<Response>
  ) => {
    try {
      const res = await fetchCall();
      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let done = false;
      let accumulatedText = "";

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        accumulatedText += decoder.decode(value || new Uint8Array(), { stream: true });

        updateTab(tabId, (t) => {
          const updated = [...t.messages];
          updated[updated.length - 1] = { role: "assistant", content: accumulatedText };
          return { ...t, messages: updated };
        });
      }
    } catch {
      updateTab(tabId, (t) => {
        const updated = [...t.messages];
        updated[updated.length - 1] = {
          role: "assistant",
          content: "Error connecting to backend.",
        };
        return { ...t, messages: updated };
      });
    }
  };

  const handleSubmit = async () => {
    const audioBlob = pendingAudioRef.current;
    if (!question.trim() && !audioBlob) return;

    const tabId = activeTabId;
    const currentImages = activeTab.images;
    const currentQuestion = question;

    // Clear state immediately
    pendingAudioRef.current = null;
    setQuestion("");
    forceUpdate((n) => n + 1);

    setLoading(true);

    let finalQuestion = currentQuestion;

    if (audioBlob) {
      // Step 1: show placeholder while transcribing
      updateTab(tabId, (t) => ({
        ...t,
        messages: [
          ...t.messages,
          { role: "user", content: "🎤 Transcribing…" },
          { role: "assistant", content: "" },
        ],
      }));

      // Step 2: transcribe audio
      try {
        const form = new FormData();
        form.append("audio", audioBlob, "recording.webm");
        const res = await fetch("http://localhost:5000/transcribe", { method: "POST", body: form });
        const { text } = await res.json();
        finalQuestion = currentQuestion ? `${text}\n${currentQuestion}` : text;
      } catch {
        finalQuestion = currentQuestion || "🎤 Voice message";
      }

      // Step 3: update user message bubble with actual transcription
      updateTab(tabId, (t) => {
        const updated = [...t.messages];
        updated[updated.length - 2] = { role: "user", content: finalQuestion };
        return { ...t, messages: updated };
      });

      // Step 4: stream LLM response with transcription as text
      await streamResponse(tabId, () =>
        fetch("http://localhost:5000/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: finalQuestion, images: currentImages }),
        })
      );
    } else {
      // Plain text: add messages and stream directly
      updateTab(tabId, (t) => ({
        ...t,
        messages: [
          ...t.messages,
          { role: "user", content: currentQuestion },
          { role: "assistant", content: "" },
        ],
      }));

      await streamResponse(tabId, () =>
        fetch("http://localhost:5000/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: currentQuestion, images: currentImages }),
        })
      );
    }

    setLoading(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleAddImage = () => {
    if (!imageInput.trim()) return;
    updateTab(activeTabId, (t) => ({
      ...t,
      images: [...t.images, imageInput.trim()],      // path/url for LLM
      imageUrls: [...t.imageUrls, imageInput.trim()], // same url for carousel
      carouselIndex: t.images.length,
    }));
    setImageInput("");
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
  const files = e.target.files;
  if (!files) return;
  for (const file of Array.from(files)) {
    const form = new FormData();
    form.append("image", file);
    try {
      const res = await fetch("http://localhost:5000/upload_image", { method: "POST", body: form });
      const { path, url } = await res.json();
      updateTab(activeTabId, (t) => ({
        ...t,
        images: [...t.images, path],
        imageUrls: [...t.imageUrls, url],
        carouselIndex: t.images.length,
      }));
    } catch {
      alert("Image upload failed.");
    }
  }
  e.target.value = "";
};

  const handleCarousel = (dir: 1 | -1) => {
    updateTab(activeTabId, (t) => ({
      ...t,
      carouselIndex: Math.max(0, Math.min(t.images.length - 1, t.carouselIndex + dir)),
    }));
  };

  const removeImage = (idx: number) => {
  updateTab(activeTabId, (t) => {
    const images = t.images.filter((_, i) => i !== idx);
    const imageUrls = t.imageUrls.filter((_, i) => i !== idx);
    return {
      ...t,
      images,
      imageUrls,
      carouselIndex: Math.min(t.carouselIndex, Math.max(0, images.length - 1)),
    };
  });
};

  const discardAudio = () => {
    pendingAudioRef.current = null;
    forceUpdate((n) => n + 1);
  };

  const toggleRecording = async () => {
    if (recording) {
      // Stop — onstop handler saves blob into pendingAudioRef
      mediaRecorderRef.current?.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      mr.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        pendingAudioRef.current = new Blob(chunksRef.current, { type: "audio/webm" });
        setRecording(false);
        forceUpdate((n) => n + 1); // show the badge
      };
      mr.start();
      mediaRecorderRef.current = mr;
      setRecording(true);
    } catch {
      alert("Microphone access denied.");
    }
  };

  const hasPendingAudio = !!pendingAudioRef.current;
  const canSend = (question.trim().length > 0 || hasPendingAudio) && !loading && !recording;

  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        overflow: "hidden",
        background: "#1c1c1e",
        fontFamily: "'SF Pro Display', 'Segoe UI', system-ui, sans-serif",
        color: "#e5e5e5",
      }}
    >
      {/* ── Tab Bar ── */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "2px",
          padding: "10px 14px 0",
          background: "#141416",
          borderBottom: "1px solid #2c2c2e",
          flexShrink: 0,
          overflowX: "auto",
        }}
      >
        {tabs.map((tab) => (
          <div
            key={tab.id}
            onClick={() => setActiveTabId(tab.id)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: "6px",
              padding: "7px 14px",
              borderRadius: "8px 8px 0 0",
              cursor: "pointer",
              fontSize: "13px",
              fontWeight: 500,
              userSelect: "none",
              whiteSpace: "nowrap",
              background: activeTabId === tab.id ? "#1c1c1e" : "transparent",
              color: activeTabId === tab.id ? "#f0f0f0" : "#888",
              borderTop: activeTabId === tab.id ? "1px solid #3a3a3c" : "1px solid transparent",
              borderLeft: activeTabId === tab.id ? "1px solid #3a3a3c" : "1px solid transparent",
              borderRight: activeTabId === tab.id ? "1px solid #3a3a3c" : "1px solid transparent",
              borderBottom: activeTabId === tab.id ? "1px solid #1c1c1e" : "none",
              transition: "all 0.15s ease",
              marginBottom: "-1px",
            }}
          >
            <span>{tab.label}</span>
            {tabs.length > 1 && (
              <span
                onClick={(e) => removeTab(tab.id, e)}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  width: "16px",
                  height: "16px",
                  borderRadius: "50%",
                  fontSize: "11px",
                  color: "#666",
                  cursor: "pointer",
                  lineHeight: 1,
                }}
                title="Close tab"
              >
                ✕
              </span>
            )}
          </div>
        ))}

        <button
          onClick={addTab}
          title="New tab"
          style={{
            marginLeft: "4px",
            width: "28px",
            height: "28px",
            borderRadius: "6px",
            border: "1px solid #3a3a3c",
            background: "transparent",
            color: "#aaa",
            fontSize: "18px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            flexShrink: 0,
            transition: "background 0.15s",
          }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#2c2c2e")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
        >
          +
        </button>
      </div>

      {/* ── Two-Column Body ── */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden", padding: "16px", gap: "16px" }}>

        {/* ── LEFT: Chat ── */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            background: "#141416",
            borderRadius: "14px",
            border: "1px solid #2c2c2e",
            overflow: "hidden",
            minWidth: 0,
          }}
        >
          {/* Messages */}
          <div
            style={{
              flex: 1,
              overflowY: "auto",
              padding: "20px 16px",
              display: "flex",
              flexDirection: "column",
              gap: "12px",
              scrollbarWidth: "thin",
              scrollbarColor: "#3a3a3c transparent",
            }}
          >
            {activeTab.messages.length === 0 && (
              <div
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#555",
                  gap: "8px",
                }}
              >
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#444" strokeWidth="1.5">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                </svg>
                <span style={{ fontSize: "14px" }}>Start a conversation</span>
              </div>
            )}

            {activeTab.messages.map((msg, i) => (
              <div
                key={i}
                style={{
                  display: "flex",
                  justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
                }}
              >
                {msg.role === "assistant" && (
                  <div
                    style={{
                      width: "28px",
                      height: "28px",
                      borderRadius: "50%",
                      background: "#2c2c2e",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      flexShrink: 0,
                      marginRight: "8px",
                      marginTop: "2px",
                      fontSize: "13px",
                    }}
                  >
                    ✦
                  </div>
                )}
                <div
                  style={{
                    maxWidth: "72%",
                    padding: "10px 14px",
                    borderRadius:
                      msg.role === "user" ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
                    background: msg.role === "user" ? "#3a3a3c" : "#212123",
                    color: "#e5e5e5",
                    fontSize: "14px",
                    lineHeight: "1.55",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                    border: msg.role === "assistant" ? "1px solid #2c2c2e" : "none",
                  }}
                >
                  {msg.content}
                  {msg.role === "assistant" && msg.content === "" && loading && (
                    <span style={{ display: "inline-flex", gap: "3px", alignItems: "center", height: "16px" }}>
                      {[0, 1, 2].map((d) => (
                        <span
                          key={d}
                          style={{
                            width: "5px",
                            height: "5px",
                            borderRadius: "50%",
                            background: "#666",
                            display: "inline-block",
                            animation: `bounce 1.2s ${d * 0.2}s infinite`,
                          }}
                        />
                      ))}
                    </span>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <div
            style={{
              padding: "12px 14px",
              borderTop: "1px solid #2c2c2e",
              background: "#141416",
              display: "flex",
              flexDirection: "column",
              gap: "8px",
            }}
          >
            {/* Pending audio badge */}
            {hasPendingAudio && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                  padding: "6px 10px",
                  background: "#1a2a1a",
                  border: "1px solid #2a3a2a",
                  borderRadius: "8px",
                  fontSize: "12px",
                  color: "#6bcf6b",
                }}
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                  <line x1="12" y1="19" x2="12" y2="23" />
                  <line x1="8" y1="23" x2="16" y2="23" />
                </svg>
                <span>Voice recording ready</span>
                <button
                  onClick={discardAudio}
                  title="Discard recording"
                  style={{
                    marginLeft: "auto",
                    background: "none",
                    border: "none",
                    color: "#6bcf6b",
                    cursor: "pointer",
                    fontSize: "13px",
                    padding: "0",
                    lineHeight: 1,
                  }}
                >
                  ✕
                </button>
              </div>
            )}

            <textarea
              placeholder={recording ? "Recording…" : "Message… (or record voice below)"}
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={recording}
              rows={2}
              style={{
                width: "100%",
                background: recording ? "#1a1a1a" : "#1c1c1e",
                border: "1px solid #2c2c2e",
                borderRadius: "10px",
                padding: "10px 12px",
                color: recording ? "#555" : "#e5e5e5",
                fontSize: "14px",
                resize: "none",
                outline: "none",
                fontFamily: "inherit",
                lineHeight: "1.5",
                boxSizing: "border-box",
                cursor: recording ? "not-allowed" : "text",
              }}
            />

            <div style={{ display: "flex", gap: "8px", alignItems: "center" }}>

              {/* Mic button */}
              <button
                onClick={toggleRecording}
                disabled={loading}
                title={recording ? "Stop recording" : "Record voice"}
                style={{
                  width: "38px",
                  height: "38px",
                  borderRadius: "10px",
                  border: `1px solid ${recording ? "#5a2a2a" : "#3a3a3c"}`,
                  background: recording ? "#3a1a1a" : "#1c1c1e",
                  color: recording ? "#ff6b6b" : "#aaa",
                  cursor: loading ? "not-allowed" : "pointer",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                  transition: "all 0.15s",
                  position: "relative",
                }}
              >
                {/* Pulsing ring while recording */}
                {recording && (
                  <span
                    style={{
                      position: "absolute",
                      inset: "-4px",
                      borderRadius: "13px",
                      border: "2px solid #ff6b6b",
                      animation: "pulse 1.2s ease-out infinite",
                      pointerEvents: "none",
                    }}
                  />
                )}
                <svg
                  width="16"
                  height="16"
                  viewBox="0 0 24 24"
                  fill={recording ? "#ff6b6b" : "none"}
                  stroke="currentColor"
                  strokeWidth="1.8"
                >
                  <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                  <line x1="12" y1="19" x2="12" y2="23" />
                  <line x1="8" y1="23" x2="16" y2="23" />
                </svg>
              </button>

              <div style={{ flex: 1 }} />

              {/* Send button */}
              <button
                onClick={handleSubmit}
                disabled={!canSend}
                style={{
                  padding: "0 20px",
                  height: "38px",
                  borderRadius: "10px",
                  border: "none",
                  background: canSend ? "#4a4a4c" : "#2c2c2e",
                  color: canSend ? "#e5e5e5" : "#555",
                  cursor: canSend ? "pointer" : "not-allowed",
                  fontSize: "13px",
                  fontWeight: 600,
                  display: "flex",
                  alignItems: "center",
                  gap: "6px",
                  transition: "all 0.15s",
                  fontFamily: "inherit",
                }}
              >
                {loading ? "Sending…" : "Send"}
                {!loading && (
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2">
                    <line x1="22" y1="2" x2="11" y2="13" />
                    <polygon points="22 2 15 22 11 13 2 9 22 2" />
                  </svg>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* ── RIGHT: Images ── */}
        <div
          style={{
            width: "500px",
            flexShrink: 0,
            display: "flex",
            flexDirection: "column",
            gap: "10px",
          }}
        >
          {/* Carousel */}
          <div
            style={{
              flex: 1,
              background: "#141416",
              borderRadius: "14px",
              border: "1px solid #2c2c2e",
              overflow: "hidden",
              display: "flex",
              flexDirection: "column",
              position: "relative",
            }}
          >
            <div
              style={{
                padding: "12px 14px 8px",
                fontSize: "11px",
                fontWeight: 600,
                letterSpacing: "0.08em",
                color: "#666",
                textTransform: "uppercase",
                borderBottom: "1px solid #2c2c2e",
                flexShrink: 0,
              }}
            >
              Images {activeTab.images.length > 0 && `· ${activeTab.carouselIndex + 1}/${activeTab.images.length}`}
            </div>

            <div
              style={{
                flex: 1,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
                overflow: "hidden",
              }}
            >
              {activeTab.images.length === 0 ? (
                <div
                  style={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    gap: "10px",
                    color: "#444",
                  }}
                >
                  <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#3a3a3c" strokeWidth="1.2">
                    <rect x="3" y="3" width="18" height="18" rx="2" />
                    <circle cx="8.5" cy="8.5" r="1.5" />
                    <polyline points="21 15 16 10 5 21" />
                  </svg>
                  <span style={{ fontSize: "13px" }}>No images</span>
                </div>
              ) : (
                <>
                  <img
                    src={activeTab.imageUrls[activeTab.carouselIndex]}   // ← was activeTab.images[...]
                    alt={`Image ${activeTab.carouselIndex + 1}`}
                    style={{
                      maxWidth: "100%",
                      maxHeight: "100%",
                      objectFit: "contain",
                      display: "block",
                    }}
                  />

                  {/* Remove button */}
                  <button
                    onClick={() => removeImage(activeTab.carouselIndex)}
                    title="Remove image"
                    style={{
                      position: "absolute",
                      top: "8px",
                      right: "8px",
                      width: "24px",
                      height: "24px",
                      borderRadius: "50%",
                      border: "none",
                      background: "rgba(0,0,0,0.6)",
                      color: "#ccc",
                      cursor: "pointer",
                      fontSize: "12px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    ✕
                  </button>

                  {/* Carousel nav */}
                  {activeTab.images.length > 1 && (
                    <>
                      <button
                        onClick={() => handleCarousel(-1)}
                        disabled={activeTab.carouselIndex === 0}
                        style={{
                          position: "absolute",
                          left: "6px",
                          width: "26px",
                          height: "26px",
                          borderRadius: "50%",
                          border: "none",
                          background: "rgba(0,0,0,0.5)",
                          color: activeTab.carouselIndex === 0 ? "#444" : "#ccc",
                          cursor: activeTab.carouselIndex === 0 ? "not-allowed" : "pointer",
                          fontSize: "14px",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        ‹
                      </button>
                      <button
                        onClick={() => handleCarousel(1)}
                        disabled={activeTab.carouselIndex === activeTab.images.length - 1}
                        style={{
                          position: "absolute",
                          right: "6px",
                          width: "26px",
                          height: "26px",
                          borderRadius: "50%",
                          border: "none",
                          background: "rgba(0,0,0,0.5)",
                          color: activeTab.carouselIndex === activeTab.images.length - 1 ? "#444" : "#ccc",
                          cursor: activeTab.carouselIndex === activeTab.images.length - 1 ? "not-allowed" : "pointer",
                          fontSize: "14px",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                        }}
                      >
                        ›
                      </button>
                    </>
                  )}

                  {/* Dots */}
                  {activeTab.images.length > 1 && (
                    <div style={{ position: "absolute", bottom: "8px", display: "flex", gap: "4px" }}>
                      {activeTab.images.map((_, i) => (
                        <div
                          key={i}
                          onClick={() => updateTab(activeTabId, (t) => ({ ...t, carouselIndex: i }))}
                          style={{
                            width: i === activeTab.carouselIndex ? "16px" : "6px",
                            height: "6px",
                            borderRadius: "3px",
                            background: i === activeTab.carouselIndex ? "#aaa" : "#444",
                            cursor: "pointer",
                            transition: "all 0.2s",
                          }}
                        />
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Image input */}
          <div
            style={{
              background: "#141416",
              borderRadius: "14px",
              border: "1px solid #2c2c2e",
              padding: "12px",
              display: "flex",
              flexDirection: "column",
              gap: "8px",
              flexShrink: 0,
            }}
          >
            <div style={{ display: "flex", gap: "6px" }}>
              <input
                type="text"
                placeholder="Paste image URL…"
                value={imageInput}
                onChange={(e) => setImageInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAddImage()}
                style={{
                  flex: 1,
                  background: "#1c1c1e",
                  border: "1px solid #2c2c2e",
                  borderRadius: "8px",
                  padding: "7px 10px",
                  color: "#e5e5e5",
                  fontSize: "12px",
                  outline: "none",
                  fontFamily: "inherit",
                  minWidth: 0,
                }}
              />
              <button
                onClick={handleAddImage}
                disabled={!imageInput.trim()}
                title="Add URL"
                style={{
                  width: "32px",
                  height: "32px",
                  borderRadius: "8px",
                  border: "1px solid #3a3a3c",
                  background: imageInput.trim() ? "#3a3a3c" : "#1c1c1e",
                  color: imageInput.trim() ? "#e5e5e5" : "#555",
                  cursor: imageInput.trim() ? "pointer" : "not-allowed",
                  fontSize: "18px",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexShrink: 0,
                  fontFamily: "inherit",
                }}
              >
                +
              </button>
            </div>

            <button
              onClick={() => fileInputRef.current?.click()}
              style={{
                width: "100%",
                padding: "7px",
                borderRadius: "8px",
                border: "1px dashed #3a3a3c",
                background: "transparent",
                color: "#777",
                cursor: "pointer",
                fontSize: "12px",
                fontFamily: "inherit",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                gap: "6px",
                transition: "all 0.15s",
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = "#1c1c1e";
                e.currentTarget.style.color = "#aaa";
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = "transparent";
                e.currentTarget.style.color = "#777";
              }}
            >
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              Upload file
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              multiple
              onChange={handleFileUpload}
              style={{ display: "none" }}
            />
          </div>
        </div>
      </div>

      <style>{`
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40% { transform: translateY(-5px); opacity: 1; }
        }
        @keyframes pulse {
          0%   { opacity: 0.8; transform: scale(1); }
          100% { opacity: 0;   transform: scale(1.35); }
        }
        textarea:focus { border-color: #4a4a4c !important; }
        input:focus    { border-color: #4a4a4c !important; }
        ::-webkit-scrollbar       { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: #3a3a3c; border-radius: 2px; }
      `}</style>
    </div>
  );
}
