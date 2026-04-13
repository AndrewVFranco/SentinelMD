import React, {useLayoutEffect, useState} from 'react';
import { Sun, Moon } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import './App.css';

function App() {
  const [query, setQuery] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [darkMode, setDarkMode] = useState(() => {
    const saved = localStorage.getItem('theme');
    return saved ? saved === 'dark' : true;
  });

  useLayoutEffect(() => {
    document.body.classList.toggle('dark-mode', darkMode);
  }, [darkMode]);

  const toggleTheme = () => {
    setDarkMode(prevMode => {
      const newMode = !prevMode;
      localStorage.setItem('theme', newMode ? 'dark' : 'light');
      return newMode;
    });
  };

  const handleSubmit = async () => {
    if (!query.trim() || loading) return;

    const userMessage = { role: 'user', text: query };
    setMessages(prev => [...prev, userMessage]);
    setQuery('');
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.text, api_key: apiKey }),
      });

      if (!response.ok) throw new Error('Query failed');
      const data = await response.json();
      const final = data.final_response;
      setResult(final);
      setMessages(prev => [...prev, { role: 'assistant', data: final }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'error',
        text: 'Failed to connect to the analysis server. Ensure the backend is running.'
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          Sentinel<span>MD</span>
        </div>

        <div>
          <button onClick={toggleTheme} style={{ background: 'none', border: 'none', outline: 'none', color: 'inherit' }}>
            {darkMode ? <Sun /> : <Moon />}
          </button>
        </div>
      </header>

      <div className="layout">
        <Sidebar apiKey={apiKey} onApiKeyChange={setApiKey} result={result}/>
        <ChatWindow
            messages={messages}
            query={query}
          onQueryChange={setQuery}
          onSubmit={handleSubmit}
          loading={loading}
        />
      </div>
    </div>
  );
}

export default App;
