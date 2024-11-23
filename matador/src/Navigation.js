import React from 'react';
import { Link } from 'react-router-dom';
import './Navigation.css';

function Navigation() {
  return (
    <nav className="navbar">
      <ul className="navbar-links">
        <li>
          <Link to="/" className="navbar-link">Home</Link>
        </li>
        <li>
          <Link to="/chat-page" className="navbar-link">Chatbot</Link>
        </li>
      </ul>
    </nav>
  );
}

export default Navigation;
