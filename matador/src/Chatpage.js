import React, { useState } from 'react';
import './Chatpage.css';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const Chatpage = () => {
    const navigate = useNavigate();
    const [menuOpen, setMenuOpen] = useState(false);
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState('');

    const toggleMenu = () => {
        setMenuOpen(!menuOpen);
    };

    const navigateToHome = () => {
        setMenuOpen(false);
        navigate('/')
    };

    const navigateToAssistant = () => {
        setMenuOpen(false);
        navigate('/chat-page');
    };

    const closeMenu = () => {
        setMenuOpen(false);
    };

    const handleSendMessage = async () => {
        
        if (inputValue.trim()) {
            const userMessage = { role: 'user', content: inputValue };
            setMessages([...messages, userMessage]);
            setInputValue('');

            try {
               
                // Send user query to the Python back-end API
                const response = await axios.post('http://localhost:5000/chat', { query: inputValue });
                
                // Append bot's response to the messages
                const botResponse = {
                    role: 'bot',
                    content: response.data.response,
                };
                setMessages(prevMessages => [...prevMessages, botResponse]);
            } catch (error) {
                // Handle any errors here
                const botResponse = {
                    role: 'bot',
                    content: "I'm sorry, there seems to be an issue with my server. Please try again later.",
                };
                setMessages(prevMessages => [...prevMessages, botResponse]);
            }
        }
    };


    return (
        <div className="chat-page">
            <div className="burger-menu" onClick={toggleMenu}>
                <div className="line"></div>
                <div className="line"></div>
                <div className="line"></div>
            </div>

            <div className={`side-menu ${menuOpen ? 'menu-open' : ''}`}>
                <div className="overlay-content">
                    <button className="mainButtonMobile" onClick={navigateToHome}>Home</button>
                    <button className="mainButtonMobile" onClick={navigateToAssistant}>Virtual Assistant</button>
                    <button className="closeButton" onClick={closeMenu}>Close</button>
                </div>
            </div>

            <div className="chat-header">
                <h1>Chat with our virtual assistant</h1>
            </div>

            <div className="chat-container">
                <div className="messages">
                    {messages.map((message, index) => (
                        <div
                            key={index}
                            className={`message ${message.role === 'user' ? 'user-message' : 'bot-message'}`}
                        >
                            {message.content}
                        </div>
                    ))}
                </div>
            </div>

            <div className="chat-input-container">
                <input
                    type="text"
                    value={inputValue}
                    placeholder="Ask me about your ideal car..."
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                />
                <button onClick={handleSendMessage}>Send</button>
            </div>
        </div>
    );
};

export default Chatpage;


