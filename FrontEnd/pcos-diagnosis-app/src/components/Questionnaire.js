import React, { useState } from 'react';

const Questionnaire = ({ onSubmit }) => {
  const [userResponses, setUserResponses] = useState({});

  const handleToggleChange = (questionId) => {
    setUserResponses((prevResponses) => ({
      ...prevResponses,
      [questionId]: !prevResponses[questionId], // Toggle the response
    }));
  };

  const handleSubmit = () => {
    onSubmit(userResponses); // Pass user responses back to the parent component
  };

  return (
    <div className="questionnaire">
      <h3>Answer the following questions to get personalized recommendations:</h3>
      {[
        'Do you have irregular periods?',
        'Do you feel fatigued or have low energy?',
        'Do you have acne?',
        'Do you have insulin resistance?',
        'Are you struggling with weight gain or weight loss?',
        'Do you have excessive hair growth (hirsutism) or androgen imbalance?',
        'Are you experiencing hair loss?',
        'Do you feel stressed or have high stress hormones?',
        'Are you trying to improve fertility?',
        'Do you have trouble sleeping?',
      ].map((question, index) => (
        <div key={index} className="question-item">
          <label>
            <input
              type="checkbox"
              checked={userResponses[index + 1] || false}
              onChange={() => handleToggleChange(index + 1)}
            />
            {question}
          </label>
        </div>
      ))}

      <button className="button" onClick={handleSubmit}>
        Submit
      </button>
    </div>
  );
};

export default Questionnaire;