import React from 'react';
import './Chatpage.css'
import Navigation from './Navigation';

const Chatpage = () => {
    return (
        <div className='background-chatpage'>
            <div className='navbar-chatpage'>
            <Navigation/>
            </div>
            <p className='start-convo-text'>Tell us about your dream car!</p>
        </div>
    );
};

export default Chatpage;