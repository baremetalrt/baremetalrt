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
      <header className="flex items-center h-16 px-6">
        <div
          className={`text-3xl font-extrabold tracking-tight ${METALLIC_GRADIENT} drop-shadow-lg flex items-center select-none`}
          style={{ fontFamily: 'Orbitron, monospace', letterSpacing: '0.08em' }}
        >
          BareMetalRT
        </div>
      </header>
      <main className="flex flex-1 flex-col items-center justify-center min-h-[70vh]">
        <div className="w-full max-w-3xl px-4 mx-auto">
          <div className="flex items-center flex-nowrap gap-2 w-full">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleInputKeyDown}
              placeholder="Enter prompt"
              className="flex-grow min-w-0 border border-gray-400 px-4 py-3 text-lg focus:outline-none focus:ring-2 focus:ring-gray-400 font-mono transition-all duration-200 focus:shadow-[0_0_0_2px_#888] hover:shadow-[0_0_0_2px_#888]"
            style={{
              background: '#222',
              color: '#eee',
              fontFamily: 'Orbitron, monospace',
              borderRadius: 0
            }}
              autoFocus
              disabled={loading}
            />
            <button
              onClick={handleSend}
              disabled={loading || !input.trim()}
              className="px-6 py-3 font-bold border border-gray-400 shadow transition-all ml-2 transform-gpu transition-transform duration-150 hover:scale-105 active:scale-95 focus:outline-none focus:ring-2 focus:ring-[#888]"
            style={{
              background: 'linear-gradient(135deg, #555 0%, #888 60%, #bdbdbd 100%)',
              color: '#222',
              fontFamily: 'Orbitron, monospace',
              letterSpacing: '0.08em',
              boxShadow: '0 2px 8px 0 rgba(90,90,90,0.22)',
              borderRadius: 0
            }}
            >
              {loading ? "..." : "Send"}
            </button>
          </div>
        </div>
        <div className="w-full max-w-3xl px-4 mt-8 flex flex-col gap-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`px-5 py-3 text-lg whitespace-pre-line self-${msg.role === "user" ? "end" : "start"} animate-fade-in-up`}
              style={msg.role === "user" ? {
                background: '#222',
                color: '#eee',
                borderRadius: 0,
                maxWidth: '80%',
                animationDelay: `${i * 60}ms`,
                animationFillMode: 'backwards'
              } : {
                background: 'linear-gradient(135deg, #555 0%, #888 60%, #bdbdbd 100%)',
                color: '#eee',
                border: '1px solid #444',
                borderRadius: 0,
                maxWidth: '80%',
                animationDelay: `${i * 60}ms`,
                animationFillMode: 'backwards'
              }}
            >
              {msg.content}
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}


