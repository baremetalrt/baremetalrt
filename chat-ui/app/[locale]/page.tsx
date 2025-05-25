"use client"

import { ChatbotUISVG } from "@/components/icons/chatbotui-svg"
import { IconArrowRight } from "@tabler/icons-react"
import { useTheme } from "next-themes"
import Link from "next/link"

import React, { useState, useRef } from "react";

const METALLIC_GRADIENT = "bg-gradient-to-r from-gray-100 via-gray-400 to-gray-200 bg-clip-text text-transparent";

export default function HomePage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<{role: "user"|"assistant", content: string}[]>([]);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMessage = { role: "user" as const, content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch("http://localhost:8000/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gpt-3.5-turbo", // or whatever your backend expects
          messages: [...messages, userMessage].map(({role, content}) => ({role, content}))
        })
      });
      const data = await res.json();
      const reply = data.choices?.[0]?.message?.content || "(No reply)";
      setMessages(prev => [...prev, { role: "assistant", content: reply }]);
    } catch (e) {
      setMessages(prev => [...prev, { role: "assistant", content: "[Error contacting backend]" }]);
    }
    setLoading(false);
    inputRef.current?.focus();
  };

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !loading) {
      handleSend();
    }
  };



  return (
    <div className="flex h-screen w-screen flex-col">
      <header className="flex items-center h-16 px-6 border-b border-gray-200 dark:border-gray-800">
        <div
          className={`text-3xl font-extrabold tracking-tight ${METALLIC_GRADIENT} drop-shadow-lg flex items-center select-none`}
          style={{ fontFamily: 'Orbitron, monospace', letterSpacing: '0.08em' }}
        >
          BareMetalRT
        </div>
      </header>
      <main className="flex flex-1 flex-col items-center justify-center min-h-[70vh]">
        <div className="w-full max-w-3xl px-4">
          <div className="flex items-center gap-2">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleInputKeyDown}
              placeholder="Send a message..."
              className="w-full border border-r-0 border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 px-4 py-3 text-lg focus:outline-none focus:ring-2 focus:ring-gray-400 font-mono"
              autoFocus
              disabled={loading}
            />
            <button
              onClick={handleSend}
              disabled={loading || !input.trim()}
              className="px-6 py-3 font-bold text-gray-700 bg-gradient-to-br from-gray-200 via-gray-400 to-gray-100 border border-gray-300 dark:border-gray-700 shadow hover:from-gray-300 hover:to-gray-300 active:scale-95 transition-all"
              style={{ fontFamily: 'Orbitron, monospace', letterSpacing: '0.08em' }}
            >
              {loading ? "..." : "Send"}
            </button>
          </div>
        </div>
        <div className="w-full max-w-3xl px-4 mt-8 flex flex-col gap-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`rounded-lg px-5 py-3 text-lg whitespace-pre-line ${msg.role === "user" ? "bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-gray-100 self-end" : "bg-gradient-to-r from-gray-200 via-gray-300 to-gray-100 text-gray-800 self-start border border-gray-300 dark:border-gray-700"}`}
              style={{ maxWidth: "80%" }}
            >
              {msg.content}
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}


