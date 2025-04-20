import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="navbar">
      <h1>PCOSight</h1>
      <div className="nav-links">
        <Link to="/">Home</Link>
        <Link to="/predict">Diagnosis</Link>
      </div>
    </nav>
  );
};

export default Navbar;