import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Homepage from './Homepage';
import Chatpage from './Chatpage';  

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Homepage />} />
        <Route path="/chat-page" element={<Chatpage />} />
      </Routes>
    </Router>
  );
};

export default App;