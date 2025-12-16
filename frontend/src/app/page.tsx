"use client";

import { useState } from "react";

const STARTER_PROMPTS = [
  "I just fired",
  "My employee asked for a raise",
  "Yesterday I turned down a candidate",
  "Here's what nobody tells you about",
  "Work-life balance is",
  "Just hired someone who",
  "Had to let go of my top performer",
  "Hustle is",
];

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [hasGenerated, setHasGenerated] = useState(false);
  const [copied, setCopied] = useState(false);

  const simulateStreaming = async (text: string) => {
    setIsStreaming(true);
    
    // Start from the current prompt
    const startLength = prompt.length;
    const words = text.slice(startLength).split(" ");
    let current = prompt;
    
    for (let i = 0; i < words.length; i++) {
      current += (i === 0 ? "" : " ") + words[i];
      setPrompt(current);
      await new Promise(resolve => setTimeout(resolve, 30 + Math.random() * 50));
    }
    
    setIsStreaming(false);
  };

  const generatePost = async () => {
    if (!prompt.trim()) return;
    
    setIsLoading(true);
    
    try {
      const response = await fetch("https://linkedin-lunatics-708213822442.asia-southeast1.run.app/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          temperature: 0.3,
          max_tokens: 100,
        }),
      });
      
      const data = await response.json();
      setHasGenerated(true);
      setIsLoading(false);
      
      // Start fake streaming into the textarea
      await simulateStreaming(data.generated_text);
    } catch (error) {
      setPrompt(prompt + " [Error: Backend not running]");
      setIsLoading(false);
      setIsStreaming(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(prompt);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const resetPost = () => {
    setPrompt("");
    setHasGenerated(false);
  };

  return (
    <main className="min-h-screen bg-[#f3f2ef]">
      {/* LinkedIn-style Header */}
      <header className="bg-white border-b border-gray-300 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-2 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <svg className="w-9 h-9 text-[#0a66c2]" viewBox="0 0 24 24" fill="currentColor">
              <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
            </svg>
            <div className="hidden md:flex items-center bg-[#eef3f8] rounded px-3 py-1.5">
              <svg className="w-4 h-4 text-gray-500 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <span className="text-gray-500 text-sm">Search</span>
            </div>
          </div>
          <nav className="flex items-center gap-6">
            <div className="flex flex-col items-center text-gray-500 hover:text-black cursor-pointer">
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M23 9v2h-2v7a3 3 0 01-3 3h-4v-6h-4v6H6a3 3 0 01-3-3v-7H1V9l11-7 11 7z"/>
              </svg>
              <span className="text-xs">Home</span>
            </div>
            <div className="flex flex-col items-center text-gray-500 hover:text-black cursor-pointer">
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 16v6H3v-6a3 3 0 013-3h3a3 3 0 013 3zm5.5-3A3.5 3.5 0 1014 9.5a3.5 3.5 0 003.5 3.5zm1 2h-2a2.5 2.5 0 00-2.5 2.5V22h7v-4.5a2.5 2.5 0 00-2.5-2.5zM7.5 2A4.5 4.5 0 1012 6.5 4.49 4.49 0 007.5 2z"/>
              </svg>
              <span className="text-xs">Network</span>
            </div>
            <div className="flex flex-col items-center text-[#0a66c2] cursor-pointer">
              <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                <path d="M21 13h-8v8h-2v-8H3v-2h8V3h2v8h8v2z"/>
              </svg>
              <span className="text-xs font-semibold">Post</span>
            </div>
          </nav>
        </div>
      </header>

      <div className="max-w-2xl mx-auto py-6 px-4">
        {/* "Premium" Banner */}
        <div className="bg-gradient-to-r from-[#f8c77e] to-[#e7a33e] rounded-lg p-4 mb-6 shadow-sm">
          <div className="flex items-center gap-3">
            <div>
              <p className="font-bold text-gray-900">Thought Leader Pro™</p>
              <p className="text-sm text-gray-700">AI-powered post generation for busy executives like Greg</p>
            </div>
          </div>
        </div>

        {/* Create Post Card */}
        <div className="bg-white rounded-lg shadow border border-gray-200 mb-4">
          <div className="p-4 border-b border-gray-200">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <img 
                  src="https://thumbs.dreamstime.com/b/confident-happy-older-business-man-ceo-leader-looking-camera-office-close-up-headshot-portrait-rich-senior-investor-381279959.jpg"
                  alt="Greg"
                  className="w-12 h-12 rounded-full object-cover"
                />
                <div>
                  <p className="font-semibold text-gray-900">Greg Hustleworth III</p>
                  <p className="text-xs text-gray-500">Post to Anyone</p>
                </div>
              </div>
              {hasGenerated && !isStreaming && (
                <button
                  onClick={copyToClipboard}
                  className="text-gray-400 hover:text-[#0a66c2] p-2 hover:bg-[#eef3f8] rounded-full transition-all"
                  title="Copy to clipboard"
                >
                  {copied ? (
                    <svg className="w-5 h-5 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  )}
                </button>
              )}
            </div>
          </div>

          <div className="p-4">
            <textarea
              value={isLoading ? prompt + " |" : prompt + (isStreaming ? "|" : "")}
              onChange={(e) => !isStreaming && !isLoading && setPrompt(e.target.value)}
              placeholder="What do you want to talk about?"
              className="w-full text-gray-800 text-lg placeholder-gray-400 resize-none focus:outline-none min-h-[200px]"
              rows={8}
              disabled={isStreaming || isLoading}
            />

            {/* Quick Prompts - only show when not generated */}
            {!hasGenerated && (
              <div className="mt-4 pt-4 border-t border-gray-100">
                <p className="text-xs text-gray-500 mb-2 font-medium">Trending thought starters:</p>
                <div className="flex flex-wrap gap-2">
                  {STARTER_PROMPTS.map((p) => (
                    <button
                      key={p}
                      onClick={() => setPrompt(p)}
                      className="px-3 py-1 bg-[#eef3f8] hover:bg-[#d0e8ff] text-[#0a66c2] rounded-full text-xs font-medium transition-all"
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Hashtags - show after generation */}
            {hasGenerated && !isStreaming && !isLoading && (
              <div className="mt-4 pt-4 border-t border-gray-100">
                <span className="text-[#0a66c2] text-sm hover:underline cursor-pointer">#Leadership</span>{" "}
                <span className="text-[#0a66c2] text-sm hover:underline cursor-pointer">#Hustle</span>{" "}
                <span className="text-[#0a66c2] text-sm hover:underline cursor-pointer">#ThoughtLeadership</span>{" "}
                <span className="text-[#0a66c2] text-sm hover:underline cursor-pointer">#Entrepreneurship</span>
              </div>
            )}
          </div>

          {/* Action Bar */}
          <div className="px-4 py-3 border-t border-gray-200 flex items-center justify-between">
            <div className="flex items-center gap-4 text-gray-500">
              <button className="flex items-center gap-1 hover:text-gray-700">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span className="text-xs hidden sm:inline">Photo</span>
              </button>
              <button className="flex items-center gap-1 hover:text-gray-700">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                </svg>
                <span className="text-xs hidden sm:inline">Video</span>
              </button>
              <button className="flex items-center gap-1 hover:text-gray-700">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span className="text-xs hidden sm:inline">Event</span>
              </button>
              {hasGenerated && !isStreaming && (
                <button 
                  onClick={resetPost}
                  className="flex items-center gap-1 hover:text-gray-700"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                  <span className="text-xs hidden sm:inline">New</span>
                </button>
              )}
            </div>
            <button
              onClick={generatePost}
              disabled={isLoading || isStreaming || !prompt.trim()}
              className="px-5 py-1.5 bg-[#0a66c2] hover:bg-[#004182] text-white font-semibold rounded-full disabled:opacity-50 disabled:cursor-not-allowed transition-all text-sm"
            >
              {isLoading ? "Generating..." : isStreaming ? "Writing..." : hasGenerated ? "Post" : "Generate"}
            </button>
          </div>
        </div>

        {/* Footer */}
        <p className="text-center text-gray-400 mt-8 text-xs">
          Powered by MLA Transformer • 114M Parameters
          <br />
          <span className="text-gray-300">This is satire. Please don't actually post these.</span>
        </p>
      </div>
    </main>
  );
}
