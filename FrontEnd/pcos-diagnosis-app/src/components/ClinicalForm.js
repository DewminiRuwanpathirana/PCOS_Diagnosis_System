import React from 'react';
import ToggleSwitch from './ToggleSwitch';

const ClinicalForm = ({ clinicalData, handleInputChange, handleToggleChange }) => {
  return (
    <div className="form-section">
      <h2>Clinical Data</h2>
      <div className="clinical-form">
        <div className="form-group">
          <label htmlFor="fsh">FSH (mIU/mL)</label>
          <input
            type="number"
            id="fsh"
            name="fsh"
            step="0.01"
            value={clinicalData.fsh}
            onChange={handleInputChange}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="lh">LH (mIU/mL)</label>
          <input
            type="number"
            id="lh"
            name="lh"
            step="0.01"
            value={clinicalData.lh}
            onChange={handleInputChange}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="age">Age (years)</label>
          <input
            type="number"
            id="age"
            name="age"
            value={clinicalData.age}
            onChange={handleInputChange}
            required
          />
        </div>

        <div className="form-group">
          <label htmlFor="cycle_length">Period Cycle Length (days)</label>
          <input
            type="number"
            id="cycle_length"
            name="cycle_length"
            value={clinicalData.cycle_length}
            onChange={handleInputChange}
            required
          />
        </div>

        <ToggleSwitch
          name="cycle"
          label="Irregular Menstrual Cycle "
          checked={clinicalData.cycle}
          onChange={handleToggleChange}
        />

        <ToggleSwitch
          name="weight_gain"
          label="Weight Gain "
          checked={clinicalData.weight_gain}
          onChange={handleToggleChange}
        />

        <ToggleSwitch
          name="hair_growth"
          label="Facial Hair Growth "
          checked={clinicalData.hair_growth}
          onChange={handleToggleChange}
        />

        <ToggleSwitch
          name="skin_darkening"
          label="Skin Darkening "
          checked={clinicalData.skin_darkening}
          onChange={handleToggleChange}
        />

        <ToggleSwitch
          name="hair_loss"
          label="Hair Loss "
          checked={clinicalData.hair_loss}
          onChange={handleToggleChange}
        />

        <ToggleSwitch
          name="pimples"
          label="Pimples "
          checked={clinicalData.pimples}
          onChange={handleToggleChange}
        />
      </div>
    </div>
  );
};

export default ClinicalForm;