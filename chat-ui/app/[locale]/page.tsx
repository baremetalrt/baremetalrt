"use client"

import { ChatbotUISVG } from "@/components/icons/chatbotui-svg"
import { IconArrowRight } from "@tabler/icons-react"
import { useTheme } from "next-themes"
import Link from "next/link"

import React, { useState, useRef } from "react";

const METALLIC_GRADIENT = "bg-gradient-to-r from-gray-100 via-gray-400 to-gray-200 bg-clip-text text-transparent";

export default function HomePage() {
  const [input, setInput] = useState("");
  const [placeholder, setPlaceholder] = useState("");

  // Typewriter animation for placeholder
  React.useEffect(() => {
    const text = "Enter prompt";
    let i = 0;
    setPlaceholder("");
    const interval = setInterval(() => {
      setPlaceholder(text.slice(0, i + 1));
      i++;
      if (i >= text.length) clearInterval(interval);
    }, 60);
    return () => clearInterval(interval);
  }, []);
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
      const res = await fetch("http://localhost:8000/v1/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: [...messages, userMessage].map(m => m.content).join("\n"),
          max_tokens: 128,
          temperature: 0.7,
          top_p: 0.95
        })
      });
      console.log('Response status:', res.status);
      const text = await res.text();
      console.log('Raw response text:', text);
      if (!res.ok) {
        setMessages(prev => [...prev, { role: "assistant", content: `[Backend error: ${res.status}] ${text}` }]);
      } else {
        let data;
        try {
          data = JSON.parse(text);
          const reply = data.choices?.[0]?.text || "(No reply)";
          setMessages(prev => [...prev, { role: "assistant", content: reply }]);
        } catch (parseErr) {
          console.error('Error parsing JSON:', parseErr);
          setMessages(prev => [...prev, { role: "assistant", content: `[Error parsing backend response]` }]);
        }
      }
    } catch (e) {
      console.error('Fetch error:', e);
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
          className={`text-2xl font-extrabold tracking-tight ${METALLIC_GRADIENT} drop-shadow-lg flex items-center select-none px-2`}
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
              placeholder={placeholder}
              className="flex-grow min-w-0 border border-gray-400 px-4 py-3 text-lg focus:outline-none focus:ring-2 focus:ring-gray-400 font-mono transition-all duration-200 focus:shadow-[0_0_0_2px_#888] hover:shadow-[0_0_0_2px_#888]"
            style={{
              background: '#222',
              color: '#eee',
              fontFamily: 'Orbitron, monospace',
              borderRadius: 0,
              '::placeholder': { color: '#bbb', opacity: 1 }
            }}
              autoFocus
              disabled={loading}
            />
            <button
              onClick={handleSend}
              disabled={loading || !input.trim()}
              aria-label="Submit prompt"
              className="ml-2 flex items-center justify-center border border-gray-400 shadow transition-all focus:outline-none focus:ring-2 focus:ring-[#888] bg-[#222] hover:bg-[#333] active:scale-95 animate-enter-btn"
              style={{
                height: '48px', // matches input height (py-3 + border)
                width: '64px',
                minWidth: '64px',
                minHeight: '48px',
                borderRadius: 0,
                boxShadow: '0 2px 8px 0 rgba(90,90,90,0.22)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(135deg, #555 0%, #888 60%, #bdbdbd 100%)',
                color: '#eee',
                fontSize: '1.3rem',
                transition: 'background 0.2s, transform 0.15s',
                cursor: loading ? 'not-allowed' : 'pointer',
              }}
            >
              {loading ? "..." : <span style={{display:'flex',alignItems:'center',justifyContent:'center',width:'100%',height:'100%'}}>&#x23CE;</span>}
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


