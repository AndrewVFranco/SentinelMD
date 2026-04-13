import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import reportWebVitals from './reportWebVitals';

const savedTheme = localStorage.getItem('theme');

if (savedTheme === 'dark' || !savedTheme) {
  document.body.classList.add('dark-mode');
} else {
  // Explicitly remove it if they saved 'light'
  document.body.classList.remove('dark-mode');
}

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();
