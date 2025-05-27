"use client"

import { ChatbotUISVG } from "@/components/icons/chatbotui-svg"
import { DevModeBanner } from "./DevModeBanner"
import { IconArrowRight } from "@tabler/icons-react"
import { useTheme } from "next-themes"
import Link from "next/link"
import { AnimatedEllipsis } from "./AnimatedEllipsis";

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
  const inputRef = useRef<HTMLTextAreaElement>(null);

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
          max_tokens: 512,
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

  const handleInputKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !loading) {
      handleSend();
    }
  };

  const activeModel = models.find(m => m.id === selectedModelId);

  return (
    <div className="flex h-screen w-screen flex-col">
      <header className="flex items-center h-16 px-6">
        <div
          className={`text-2xl font-extrabold tracking-tight ${METALLIC_GRADIENT} drop-shadow-lg flex items-center select-none px-2`}
          style={{ fontFamily: 'Orbitron, monospace', letterSpacing: '0.08em' }}
        >
          BareMetalRT
          {process.env.NODE_ENV === 'development' && <span style={{color:'#c0c0c0'}}><DevModeBanner /></span>}
        </div>
      </header>
      {/* Model selector UI */}
      <div className="w-full max-w-3xl px-4 mx-auto mt-4 mb-2">
        <div className="flex justify-center gap-2 items-center">
          {models.length === 0 ? (
            <span className="text-gray-400 searching-ellipsis">Searching for models<span className="ellipsis">…</span></span>
          ) : (
            <div className="flex gap-3">
              {models.map(model => (
                <button
                  key={model.id}
                  disabled={model.status === 'offline'}
                  onClick={() => handleModelSelect(model.id)}
                  title={model.status === 'offline' ? `${model.name} (Offline)` : model.description}
                  className={`model-btn${selectedModelId === model.id ? ' selected' : ''}${model.status === 'offline' ? ' offline-btn' : ''}${model.name.toLowerCase().includes('llama 2 70b') && selectedModelId !== model.id && model.status !== 'offline' ? ' llama-btn' : ''}`}
                  style={{
                    opacity: model.status === 'offline' ? 0.7 : 1,
                    cursor: model.status === 'offline' ? 'not-allowed' : 'pointer',
                    background: model.status === 'offline' ? '#181A1B' : (model.name.toLowerCase() === 'petals' && selectedModelId !== model.id ? '#181A1B' : (model.name.toLowerCase().includes('llama 2 70b') && selectedModelId !== model.id ? '#232428' : '#222')),
                    color: model.status === 'offline' ? '#888' : (model.name.toLowerCase().includes('llama 2 70b') && selectedModelId !== model.id ? '#e8e9ea' : '#fff'),
                    border: model.status === 'offline' ? '1.5px solid #232428' : '1px solid #bdbdbd',
                    borderRadius: 0,
                    padding: '2px 7px',
                    fontSize: '0.83rem',
                    fontWeight: 500,
                    fontFamily: 'Orbitron, monospace',
                    marginRight: 0,
                    minWidth: 0,
                    animation: 'fadeScaleIn 0.4s cubic-bezier(.35,1.6,.6,1)'
                  }}
                >
                  {(() => {
                    const match = model.name.match(/^(.*?)(\s*\(.*\))$/);
                    if (match) {
                      return <span style={{display:'inline-block',textAlign:'center'}}>
                        {match[1].trim()}<br/>{match[2].trim()}
                      </span>;
                    }
                    return model.name;
                  })()}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <main className="flex flex-1 flex-col items-center justify-center min-h-[70vh]">

        {/* Model status bar */}
        <div className="w-full max-w-3xl px-4 mx-auto mt-4 mb-2">
          <div className="flex flex-col gap-2">
            {firstLoad && modelsLoading ? (
              <span className="text-gray-400 font-mono text-sm">Checking model status...</span>
            ) : models.length === 0 ? (
              <span className="text-red-500 font-mono text-sm">No models online</span>
            ) : (
              activeModel ? (
                <div key={activeModel.id} className="flex items-center gap-2 font-mono text-sm py-1 px-2 rounded bg-black/10">
                  <span>{activeModel.name}</span>
                  <span className={"ml-2 font-bold online-pulse" + (activeModel.status === 'online' ? '' : ' text-gray-400')}>
                    {activeModel.status === 'online' ? 'Online' : 'Offline'}
                  </span>
                </div>
              ) : (
                <span className="text-gray-400 font-mono text-sm pulse-select">Select a model</span>
              )
            )}
          </div>
        {/* Model button animations and status pulses */}
      <style>{`

          0% { opacity: 0; transform: scale(0.7); }
          80% { opacity: 1; transform: scale(1.05); }
          100% { opacity: 1; transform: scale(1); }
        }
        @keyframes model-btn-pulse {
  0% {
    box-shadow:
      0 0 0 0 #f8f8fa88,
      0 0 0 0 #c0c0c0bb,
      0 0 0 0 #bdbdbd44;
    transform: scale(1);
  }
  50% {
    box-shadow:
      0 0 12px 6px #f8f8fa88,
      0 0 24px 12px #c0c0c0bb,
      0 0 32px 18px #bdbdbd44;
    transform: scale(1.045);
  }
  100% {
    box-shadow:
      0 0 0 0 #f8f8fa88,
      0 0 0 0 #c0c0c0bb,
      0 0 0 0 #bdbdbd44;
    transform: scale(1);
  }
}
        @keyframes online-green-pulse {
          0% { color: #27c93f; text-shadow: 0 0 4px #27c93f66, 0 0 12px #27c93f22; }
          50% { color: #27c93f; text-shadow: 0 0 16px #27c93fcc, 0 0 32px #27c93f77; }
          100% { color: #27c93f; text-shadow: 0 0 4px #27c93f66, 0 0 12px #27c93f22; }
        }
        @keyframes pulse-select {
            0% { opacity: 0.85; text-shadow: 0 0 0.5px #bdbdbd33, 0 0 1px #c0c0c011; }
            50% { opacity: 0.55; text-shadow: 0 0 1.5px #bdbdbd22, 0 0 3px #c0c0c022; }
            100% { opacity: 0.85; text-shadow: 0 0 0.5px #bdbdbd33, 0 0 1px #c0c0c011; }
          }
          .pulse-select {
            animation: pulse-select 3.5s infinite;
          }
        .model-btn {
          transition: background 0.22s, color 0.22s, transform 0.18s cubic-bezier(.35,1.6,.6,1), box-shadow 0.18s;
          background: #222;
          color: #fff;
          font-weight: 600;
          letter-spacing: 0.01em;
        }
        .model-btn:hover:not(:disabled), .model-btn:focus-visible:not(:disabled) {
          background: #222 !important;
          color: #fff !important;
          transform: scale(1.08);
          box-shadow: none;
        }
        .model-btn.offline-btn {
          background: #181A1B !important;
          color: #888 !important;
          border: 1.5px solid #232428 !important;
          font-weight: 600;
          filter: grayscale(0.2) brightness(1.0);
        }
        .model-btn.selected {
  background: linear-gradient(90deg, #f8f8fa 0%, #c0c0c0 50%, #bdbdbd 100%) !important;
  color: #222 !important;
  border: 2px solid #bdbdbd !important;
  animation: model-btn-pulse 1.5s infinite;
  box-shadow: 0 0 0 3px #f8f8fa99, 0 0 16px 4px #c0c0c0cc;
}
        .model-btn:active:not(:disabled) {
          transform: scale(0.96);
        }
        .online-pulse {
          animation: online-silver-pulse 1.1s infinite;
        }
      `}</style>
        </div>
        <div className="w-full max-w-3xl px-4 mx-auto">
          <div className="flex items-end flex-nowrap gap-2 w-full">
            <textarea
              ref={inputRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleInputKeyDown}
              placeholder={placeholder}
              rows={3}
              className="flex-grow min-w-0 border border-gray-400 px-4 py-3 focus:outline-none focus:ring-2 focus:ring-gray-400 font-mono transition-all duration-200 focus:shadow-[0_0_0_2px_#888] hover:shadow-[0_0_0_2px_#888] resize-none"
              style={{
                background: '#222',
                color: '#eee',
                fontFamily: 'Orbitron, monospace',
                borderRadius: 0,
                fontSize: '1rem',
                alignSelf: 'stretch',
                marginBottom: 0
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
                alignSelf: 'flex-end',
                width: '64px',
                minWidth: '64px',
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
                marginBottom: 0
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
        <div className="w-full max-w-3xl px-4 mt-8 flex flex-col gap-4" style={{ fontSize: '1rem', lineHeight: 1.6 }}>
          {/* Chat message area with scroll and max height to prevent clipping into prompt */}
          <div style={{
            maxHeight: '45vh', // or 340px if you prefer a fixed px value
            minHeight: '120px',
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '1rem',
          }}>
            {(() => {
              const [showAllMessages, setShowAllMessages] = React.useState(false);
              const visibleMessages = showAllMessages || messages.length <= 3 ? messages : messages.slice(-3);
              return <>
                {messages.length > 3 && !showAllMessages && (
                  <button
                    className="mb-2 text-xs px-3 py-1 bg-[#232428] text-gray-200 hover:bg-[#333] border border-gray-500"
                    style={{ alignSelf: 'center', borderRadius: 0 }}
                    onClick={() => setShowAllMessages(true)}
                  >
                    Load previous messages
                  </button>
                )}
                {visibleMessages.map((msg, idx) => (
                  <div
                    key={showAllMessages ? idx : messages.length - visibleMessages.length + idx}
                    className={`px-5 py-3 whitespace-pre-line self-${msg.role === "user" ? "end" : "start"} animate-fade-in-up`}
                    style={{
                      fontSize: '1rem',
                      lineHeight: 1.6,
                      background: msg.role === "user" ? '#222' : '#e5e7eb',
                      color: msg.role === "user" ? '#eee' : '#222',
                      border: msg.role === "user" ? undefined : '1px solid #bdbdbd',
                      borderRadius: 0,
                      maxWidth: '80%',
                      maxHeight: 300,
                      overflowY: 'auto',
                      animationDelay: `${idx * 60}ms`,
                      animationFillMode: 'backwards',
                      wordBreak: 'break-word',
                    }}
                  >
                    {msg.content}
                  </div>
                ))}
              </>;
            })()}
          </div>
        </div>
      </main>
    </div>
  );
}
