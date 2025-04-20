import React from 'react';
import { useNavigate } from 'react-router-dom';
import pcos3Image from '../assets/pcos4.jpg'; // Long image for the top section

const Home = () => {
  const navigate = useNavigate();

  return (
    <div className="container">
      <div className="content-card">
        {/* Header */}
        <header className="header">
          <h1>Welcome to The PCOS Diagnosis System</h1>
          <p className="subtitle">Empowering Women’s Health with Cutting-Edge Imaging and Clinical Precision</p>
        </header>

        {/* Long Image */}
        <div className="long-image-container">
          <img src={pcos3Image} alt="PCOS Diagnosis" className="long-image" />
        </div>
        
        {/* Full-width Intro Section */}
        <div className="full-width-section">
          <section className="intro-section">
            <h2 className="section-title">Advanced PCOS Diagnostic Tool</h2>
            <p className="intro-text">
              Welcome to our Polycystic Ovary Syndrome (PCOS) diagnosis platform, 
              designed to deliver accurate and evidence-based diagnostic results by integrating 
              clinical data analysis with advanced ultrasound imaging technology.
            </p>
          </section>
        </div>

        <div className="content-divider"></div>

        {/* Process and Criteria Wrapper */}
        <div className="process-criteria-wrapper">
          <section className="process-section">
            <h3 className="subsection-title">Streamlined Diagnostic Process</h3>
            <div className="steps-container">
              <div className="step-item">
                <div className="step-number">1</div>
                <div className="step-content">
                  <h4>Clinical Assessment</h4>
                  <p>Enter patient data including menstrual history, hormonal profiles, and clinical symptoms.</p>
                </div>
              </div>
              
              <div className="step-item">
                <div className="step-number">2</div>
                <div className="step-content">
                  <h4>Image Analysis</h4>
                  <p>Upload ultrasound images for automated detection of polycystic ovarian cysts.</p>
                </div>
              </div>
              
              <div className="step-item">
                <div className="step-number">3</div>
                <div className="step-content">
                  <h4>Comprehensive Results</h4>
                  <p>Receive detailed diagnostic analysis based on the Rotterdam Criteria with clinical recommendations.</p>
                </div>
              </div>
            </div>
          </section>

          <section className="criteria-section">
            <h3 className="subsection-title">Rotterdam Diagnostic Criteria</h3>
            <p>
              Our system diagnoses PCOS using internationally recognized Rotterdam Criteria, 
              which identifies PCOS when at least two of these three key indicators are present:
            </p>
            <div className="criteria-container">
              <div className="criteria-item">
                <div className="criteria-marker"></div>
                <p>Oligo-ovulation or anovulation (irregular or absent menstrual cycles).</p>
              </div>
              <div className="criteria-item">
                <div className="criteria-marker"></div>
                <p>Clinical and/or biochemical signs of hyperandrogenism (High level of androgen).</p>
              </div>
              <div className="criteria-item">
                <div className="criteria-marker"></div>
                <p>Polycystic ovaries on ultrasound examination.</p>
              </div>
            </div>
          </section>
        </div>

        {/* Button */}
        <div className="cta-container">
          <button className="cta-button" onClick={() => navigate('/predict')}>
            Begin Assessment
          </button>
        </div>
      </div>
    </div>
  );
};

export default Home;