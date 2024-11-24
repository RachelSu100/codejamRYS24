import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Navigation.css';

function Navigation() {
    const navigate = useNavigate();
  
    return (
      <nav className="navbar">
          <button className="navbar-button" onClick={() => navigate('/')}>Home</button>
          <button className="navbar-button" onClick={() => navigate('/chat-page')}>Chatbot</button>
      </nav>
    );
  }

export default Navigation;
