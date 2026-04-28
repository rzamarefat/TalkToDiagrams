"use client";

import { useState } from "react";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [image, setImage] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!question) return;

    setLoading(true);
    setResponse("");

    try {
      const res = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          images: image ? [image] : [],
        }),
      });

      if (!res.body) {
        throw new Error("No response body");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");

      let done = false;
      let accumulatedText = "";

      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;

        const chunk = decoder.decode(value || new Uint8Array(), {
          stream: true,
        });

        accumulatedText += chunk;

        // Live update UI
        setResponse(accumulatedText);
      }
    } catch (error) {
      console.error("Error:", error);
      setResponse("Error connecting to backend.");
    }

    setLoading(false);
  };

  return (
    <div style={{ padding: "40px", fontFamily: "Arial" }}>
      <h1>Flask Analyzer UI (Streaming)</h1>

      <textarea
        placeholder="Enter your question..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        rows={4}
        style={{
          width: "100%",
          marginBottom: "10px",
          padding: "10px",
        }}
      />

      <input
        type="text"
        placeholder="Image path or URL"
        value={image}
        onChange={(e) => setImage(e.target.value)}
        style={{
          width: "100%",
          marginBottom: "10px",
          padding: "10px",
        }}
      />

      <button
        onClick={handleSubmit}
        disabled={loading}
        style={{
          padding: "10px 20px",
          cursor: loading ? "not-allowed" : "pointer",
        }}
      >
        {loading ? "Analyzing..." : "Analyze"}
      </button>

      <hr style={{ margin: "20px 0" }} />

      <h3>Response:</h3>

      <div
        style={{
          whiteSpace: "pre-wrap",
          background: "#f5f5f5",
          padding: "15px",
          borderRadius: "5px",
          minHeight: "100px",
        }}
      >
        {response || "No response yet..."}
      </div>
    </div>
  );
}