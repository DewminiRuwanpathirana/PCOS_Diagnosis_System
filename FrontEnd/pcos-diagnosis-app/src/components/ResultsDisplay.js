import React from 'react';

const ResultsDisplay = ({ results, isLoading }) => {
  if (isLoading) {
    return <div className="loading">Processing...</div>;
  }

  if (!results) {
    return null;
  }

  const { clinical_prediction, image_prediction, rotterdam_criteria_result } = results;

  // Handle safe access to probabilities
  const clinicalProbability = clinical_prediction?.probabilities?.PCOS ||
    clinical_prediction?.probabilities?.['NO PCOS'] || 0;

  const imageProbability = image_prediction?.probabilities?.PCOS ||
    image_prediction?.probabilities?.['NO PCOS'] || 0;

  return (
    <div className="results">
      <h2>Results</h2>
      
      <div className="prediction">
        Clinical Analysis: {clinical_prediction?.prediction || 'N/A'}
      </div>
      
      {clinical_prediction?.shap_plot?.PCOS && (
        <div id="shap-analysis">
          <h3>Feature Importance Analysis</h3>
          <img 
            className="shap-plot" 
            src={`data:image/png;base64,${clinical_prediction.shap_plot.PCOS}`} 
            alt="SHAP Plot" 
          />
        </div>
      )}
      
      <div className="prediction">
        Image Analysis: {image_prediction?.prediction || 'N/A'}
      </div>
      
      {image_prediction?.visualization?.PCOS && (
        <div id="heatmap-container">
          <h3>Image Analysis Visualization</h3>
          <img 
            className="heatmap" 
            src={`data:image/jpeg;base64,${image_prediction.visualization.PCOS}`} 
            alt="Heatmap Visualization" 
          />
        </div>
      )}
      
      {rotterdam_criteria_result && (
        <div className="prediction rotterdam-result">
          <strong>Final Prediction according to the Rotterdam Criteria:</strong>{' '}
          {rotterdam_criteria_result['Final Prediction according to Rotterdam criteria']}<br />
          Confidence: {((clinicalProbability * 100 + imageProbability * 100 + 100) / 3).toFixed(2)}%
        </div>
      )}
    </div>
  );
};

export default ResultsDisplay;