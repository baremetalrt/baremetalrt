"use client"

import { ChatbotUISVG } from "@/components/icons/chatbotui-svg"
import { DevModeBanner } from "./DevModeBanner"
import { IconArrowRight } from "@tabler/icons-react"
import { useTheme } from "next-themes"
import Link from "next/link"

import React, { useState, useRef } from "react";

const METALLIC_GRADIENT = "bg-gradient-to-r from-gray-100 via-gray-400 to-gray-200 bg-clip-text text-transparent";

import { API_URL } from "@/lib/api";

export default function HomePage() {
  const [models, setModels] = useState<{id:string, name:string, status:string, description:string}[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [modelsLoading, setModelsLoading] = useState(true);
  const [firstLoad, setFirstLoad] = useState(true);
  const fetchModels = async () => {
    if (firstLoad) setModelsLoading(true);
    try {
      const res = await fetch(`${API_URL}/api/models`);
      if (!res.ok) throw new Error('Failed to fetch models');
      const data = await res.json();
      setModels(data);
    } catch {
      setModels([]); // treat as offline
    }
    setModelsLoading(false);
    setFirstLoad(false);
  };
  React.useEffect(() => {
    fetchModels();
    const interval = setInterval(fetchModels, 10000); // poll every 10s
    return () => clearInterval(interval);
  }, []);

  // Model switch handler
  const handleModelSelect = async (modelId: string) => {
    if (models.find(m => m.id === modelId && m.status === 'online')) {
      setSelectedModelId(modelId);
      try {
        await fetch(`${API_URL}/api/switch_model`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model_id: modelId })
        });
      } catch (e) {
        // Optionally show error
      }
    }
  };

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
      const res = await fetch(`${API_URL}/v1/completions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: userMessage.content,
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
          {process.env.NODE_ENV === 'development' && <DevModeBanner />}
        </div>
      </header>
      <main className="flex flex-1 flex-col items-center justify-center min-h-[70vh]">
        {/* Model selector UI */}
        <div className="w-full max-w-3xl px-4 mx-auto mt-4 mb-2">
          <div className="flex gap-4 items-center">
            <span className="font-mono font-bold text-lg">Model:</span>
            {models.length === 0 ? (
              <span className="text-gray-400">No models found</span>
            ) : (
              <div className="flex gap-2">
                {models.map(model => (
                  <button
                    key={model.id}
                    disabled={model.status === 'offline'}
                    onClick={() => handleModelSelect(model.id)}
                    title={model.description}
                    style={{
                      opacity: model.status === 'offline' ? 0.5 : 1,
                      cursor: model.status === 'offline' ? 'not-allowed' : 'pointer',
                      background: selectedModelId === model.id ? '#222' : '#e5e7eb',
                      color: selectedModelId === model.id ? '#fff' : '#222',
                      border: '1px solid #bdbdbd',
                      borderRadius: 4,
                      padding: '8px 16px',
                      fontWeight: 600,
                      fontFamily: 'Orbitron, monospace',
                      marginRight: 8
                    }}
                  >
                    {model.name}
                    {model.status === 'offline' && (
                      <span style={{ marginLeft: 8, color: '#e53e3e', fontWeight: 'bold', fontSize: 12 }}>OFFLINE</span>
                    )}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>
        {/* Model status bar */}
        <div className="w-full max-w-3xl px-4 mx-auto mt-4 mb-2">
          <div className="flex items-center gap-3">
            {firstLoad && modelsLoading ? (
              <span className="text-gray-400 font-mono text-sm">Checking model status...</span>
            ) : models.length === 0 ? (
              <span className="text-red-500 font-mono text-sm">No models online</span>
            ) : models.map(model => (
              <span key={model.id} className="flex items-center gap-1 font-mono text-sm">
                <span
                  style={{
                    display: 'inline-block',
                    width: 10,
                    height: 10,
                    borderRadius: '50%',
                    marginRight: 6,
                    background: model.status === 'online' ? '#27c93f' : '#888',
                    border: model.status === 'online' ? '2px solid #27c93f' : '2px solid #aaa',
                    boxShadow: model.status === 'online' ? '0 0 8px #27c93f, 0 0 0 #27c93f' : undefined,
                    transition: 'background 0.4s, border 0.4s, box-shadow 0.4s',
                    animation: model.status === 'online' ? 'pulse-dot 1.3s infinite ease-in-out' : undefined
                  }}
                ></span>
                {model.name}
                <span className="ml-2 text-gray-400">{model.status === 'online' ? 'Online' : 'Offline'}</span>
              </span>
            ))}
          </div>
        {/* Pulse animation style */}
        <style>{`
          @keyframes pulse-dot {
            0% { box-shadow: 0 0 8px #27c93f, 0 0 0 #27c93f; }
            50% { box-shadow: 0 0 16px #27c93f, 0 0 8px #27c93f33; }
            100% { box-shadow: 0 0 8px #27c93f, 0 0 0 #27c93f; }
          }
        `}</style>
        </div>
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
                borderRadius: 0
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
        {/* Generating response animation */}
        {loading && (
          <div className="w-full max-w-3xl px-4 mt-2 mb-2">
            <div
              className="text-sm font-mono mb-1"
              style={{
                background: 'linear-gradient(90deg, #bdbdbd 0%, #e5e7eb 50%, #bdbdbd 100%)',
                backgroundSize: '200% 100%',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                animation: 'shimmer 1.2s linear infinite',
                display: 'inline-block',
              }}
            >
              Generating response...
            </div>
            <style>{`
              @keyframes shimmer {
                0% { background-position: -200% 0; }
                100% { background-position: 200% 0; }
              }
            `}</style>
          </div>
        )}
        <div className="w-full max-w-3xl px-4 mt-8 flex flex-col gap-4">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`px-5 py-3 text-lg whitespace-pre-line self-${msg.role === "user" ? "end" : "start"} animate-fade-in-up`}
              style={msg.role === "user"
  ? {
      background: '#222',
      color: '#eee',
      borderRadius: 0,
      maxWidth: '80%',
      animationDelay: `${i * 60}ms`,
      animationFillMode: 'backwards'
    }
  : {
      background: '#e5e7eb',
      color: '#222',
      border: '1px solid #bdbdbd',
      borderRadius: 0,
      maxWidth: '80%',
      animationDelay: `${i * 60}ms`,
      animationFillMode: 'backwards'
    }
}
            >
              {msg.content}
            </div>
          ))}
        </div>
      </main>
    </div>
  );
}
