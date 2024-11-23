import React from 'react';
import './Homepage.css';
import cars from './resources/cars.mp4';
import { useNavigate } from 'react-router-dom';
import Navigation from './Navigation';
import 'bootstrap/dist/css/bootstrap.min.css';

const Homepage = () => {
    const navigate = useNavigate();

    const handleClick = () => {
        navigate('/chat-page');
    };

    return (
        <main className="main-container">
            <div className='video-container'>
                <video className='background-video' autoPlay loop muted>
                    <source src={cars} type='video/mp4' />
                </video>
            </div>

            <div className='container'>
                <div>
                    <Navigation/>
                </div>
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