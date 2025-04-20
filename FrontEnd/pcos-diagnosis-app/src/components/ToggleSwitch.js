import React from 'react';

const ToggleSwitch = ({ name, label, checked, onChange }) => {
  return (
    <div className="form-group">
      <label className="toggle-label">
        {label}
        <div className="toggle-switch">
          <input
            type="checkbox"
            name={name}
            checked={checked}
            onChange={onChange}
          />
          <span className="toggle-slider"></span>
        </div>
      </label>
    </div>
  );
};

export default ToggleSwitch;