import React, { useState } from 'react';
import './Homepage.css';
import cars from './resources/cars.mp4';
import { useNavigate } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';

const Homepage = () => {
    const navigate = useNavigate();
    const [menuOpen, setMenuOpen] = useState(false);

    const handleClick = () => {
        navigate('/chat-page');
    };

    const toggleMenu = () => {
        setMenuOpen(!menuOpen);
    };

    const navigateToHome = () => {
        setMenuOpen(false);
    };

    const navigateToAssistant = () => {
        setMenuOpen(false);
        navigate('/chat-page');
    };

    const closeMenu = () => {
        setMenuOpen(false);
    };


    return (
        <main className="main-container">
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

            <div className='video-container'>
                <video className='background-video' autoPlay loop muted>
                    <source src={cars} type='video/mp4' />
                </video>
            </div>

            <div className='container'>
                <div className='text-container'>
                <p className="tagline-text">Find your dream car.</p>
                <button 
                    className="get-started-button" 
                    onClick={handleClick}
                >
                    Get started now
                </button>
                </div>
            </div>

        </main>
    );
};

export default Homepage;