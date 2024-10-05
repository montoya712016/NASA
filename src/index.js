import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import App from './App'; // Sua página principal
import Health from './Health'; // Outra página
import Home from './Home'; // Outra página
import './index.css'; // Arquivo de estilos com Tailwind
import Faq from './Faq';

const root = ReactDOM.createRoot(document.getElementById('root'));

root.render(
  <Router>
    <Routes>
    <Route path="/" element={<Home />} />
      <Route path="/dashboard" element={<App />} />
      <Route path="/health" element={<Health />} />
      <Route path="/faq" element={<Faq />} />
    </Routes>
  </Router>
);