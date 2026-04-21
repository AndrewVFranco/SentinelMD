import React, {useLayoutEffect, useState} from 'react';
import { Sun, Moon } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

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

  const handleSubmit = async (fhirData = {}) => {
    if ((!query.trim() && !fhirData.has_fhir) || loading) return;

    const userMessage = {
        role: 'user',
        text: query || `FHIR ${fhirData.fhir_resource_type}/${fhirData.fhir_resource_id}`
      };
    setMessages(prev => [...prev, userMessage]);
    setQuery('');
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.text, api_key: apiKey, ...fhirData }),
      });

      if (!response.ok) throw new Error('Query failed');
      const data = await response.json();
      console.log('full response:', data);
      console.log('final_response:', data.final_response);
      const final = data.final_response;
      setResult(final);
      setMessages(prev => [...prev, { role: 'assistant', data: final }]);
    } catch (err) {
        let errorText = 'Failed to connect to the analysis server. Ensure the backend is running.';

        if (err.message === 'Query failed') {
          errorText = 'The server returned an error. Check your API key or try a different query.';
        } else if (err.message.includes('fetch')) {
          errorText = 'Cannot reach the backend server. Ensure it is running on port 8000.';
        } else if (err.message.includes('quota') || err.message.includes('429')) {
          errorText = 'API quota exhausted. Please provide your own Gemini API key.';
        }

        setMessages(prev => [...prev, {
          role: 'error',
          text: errorText
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
