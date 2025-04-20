import React, { useState } from 'react';
import ClinicalForm from '../components/ClinicalForm';
import ImageUpload from '../components/ImageUpload';
import ResultsDisplay from '../components/ResultsDisplay';
import Questionnaire from '../components/Questionnaire';

const Prediction = () => {
  const [clinicalData, setClinicalData] = useState({
    fsh: '', lh: '', age: '', cycle_length: '',
    weight_gain: false, hair_growth: false, skin_darkening: false,
    hair_loss: false, pimples: false, cycle: false,
  });

  const [imageFile, setImageFile] = useState(null);
  const [results, setResults] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showQuestionnaire, setShowQuestionnaire] = useState(false);
  const [userResponses, setUserResponses] = useState({});
  const [resetImage, setResetImage] = useState(false); // State to reset image preview

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setClinicalData({ ...clinicalData, [name]: value });
  };

  const handleToggleChange = (e) => {
    const { name, checked } = e.target;
    setClinicalData({ ...clinicalData, [name]: checked });
  };

  const handleImageChange = (e) => {
    setImageFile(e.target.files[0]);
    setResetImage(false); // Reset the resetImage flag when a new image is uploaded
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setIsLoading(true);

    try {
      if (!imageFile) throw new Error('Please select an ultrasound image');

      const formattedClinicalData = {
        'FSH(mIU/mL) ': parseFloat(clinicalData.fsh),
        'LH(mIU/mL) ': parseFloat(clinicalData.lh),
        ' Age (yrs) ': parseInt(clinicalData.age),
        'Cycle length(days) ': parseInt(clinicalData.cycle_length),
        'Weight gain(Y/N) ': clinicalData.weight_gain ? 1 : 0,
        'hair growth(Y/N) ': clinicalData.hair_growth ? 1 : 0,
        'Skin darkening (Y/N) ': clinicalData.skin_darkening ? 1 : 0,
        'Hair loss(Y/N) ': clinicalData.hair_loss ? 1 : 0,
        'Pimples(Y/N) ': clinicalData.pimples ? 1 : 0,
        'Cycle(R/I) ': clinicalData.cycle ? 4 : 2,
      };

      const formData = new FormData();
      formData.append('clinical_data', JSON.stringify(formattedClinicalData));
      formData.append('image', imageFile);

      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      setResults(data);
    } catch (err) {
      setError(err.message || 'An unknown error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRestartDiagnosis = () => {
    // Reset all states to restart the diagnosis
    setClinicalData({
      fsh: '', lh: '', age: '', cycle_length: '',
      weight_gain: false, hair_growth: false, skin_darkening: false,
      hair_loss: false, pimples: false, cycle: false,
    });
    setImageFile(null);
    setResults(null);
    setUserResponses({});
    setShowQuestionnaire(false);
    setResetImage(true); // Trigger image preview reset
  };

  const handleQuestionnaireSubmit = (responses) => {
    setUserResponses(responses);
    setShowQuestionnaire(false);
  };

  const generateRecommendations = () => {
    const questions = [
      { id: 1, symptom: 'Do you have irregular periods?', supplement: 'Inositol' },
      { id: 2, symptom: 'Do you feel fatigued or have low energy?', supplement: 'Fish Oil' },
      { id: 3, symptom: 'Do you have acne?', supplement: 'Zinc, Fish Oil' },
      { id: 4, symptom: 'Do you have insulin resistance?', supplement: 'Inositol' },
      { id: 5, symptom: 'Are you struggling with weight gain or weight loss?', supplement: 'Green Tea, Omega-3, Inositol' },
      { id: 6, symptom: 'Do you have excessive hair growth (hirsutism) or androgen imbalance?', supplement: 'Spearmint Tea, Zinc' },
      { id: 7, symptom: 'Are you experiencing hair loss?', supplement: 'Rosemary Oil, Fish Oil' },
      { id: 8, symptom: 'Do you feel stressed or have high stress hormones?', supplement: 'Magnesium' },
      { id: 9, symptom: 'Are you trying to improve fertility?', supplement: 'Vitamin D' },
      { id: 10, symptom: 'Do you have trouble sleeping?', supplement: 'Magnesium' },
    ];

    return questions
      .filter((question) => userResponses[question.id])
      .map((question) => (
        <div key={question.id} className="recommendation-item">
          <p><strong>{question.symptom}</strong></p>
          <p>➡ {question.supplement}</p>
        </div>
      ));
  };

  return (
    <div className="prediction-page">
      <form onSubmit={handleSubmit}>
        <ClinicalForm
          clinicalData={clinicalData}
          handleInputChange={handleInputChange}
          handleToggleChange={handleToggleChange}
        />
        <ImageUpload handleImageChange={handleImageChange} resetImage={resetImage} />
        <button type="submit" className="button">Analyze</button>
      </form>

      {error && <div className="error">{error}</div>}

      <ResultsDisplay results={results} isLoading={isLoading} />

      {results && !showQuestionnaire && (
        <div className="action-buttons">
          <button className="button" onClick={() => setShowQuestionnaire(true)}>
            Get Self-Treatment Recommendations
          </button>
          <button className="button secondary" onClick={handleRestartDiagnosis}>
            Diagnose Again
          </button>
        </div>
      )}

      {showQuestionnaire && <Questionnaire onSubmit={handleQuestionnaireSubmit} />}

      {Object.keys(userResponses).length > 0 && (
        <div className="recommendations">
          <h3>Personalized Self-Treatment Recommendations:</h3>
          {generateRecommendations()}
          <p>
            These supplements are commonly recommended for managing PCOS symptoms.
          </p>
        </div>
      )}
    </div>
  );
};

export default Prediction;