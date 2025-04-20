import React, { useState, useEffect } from 'react';

const ImageUpload = ({ handleImageChange, resetImage }) => {
  const [imagePreview, setImagePreview] = useState(null);

  // Reset image preview when resetImage prop changes
  useEffect(() => {
    if (resetImage) {
      setImagePreview(null);
    }
  }, [resetImage]);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImagePreview(reader.result); // Set the image preview URL
      };
      reader.readAsDataURL(file); // Read the file as a data URL
    }
    handleImageChange(e); // Call the parent handler
  };

  return (
    <div className="form-section">
      <h2>Upload Ultrasound Scan Image</h2>
      <input
        type="file"
        name="image"
        accept="image/*"
        onChange={handleImageUpload}
        required
      />
      {/* Display the image preview */}
      {imagePreview && (
        <div className="image-preview">
          <img src={imagePreview} alt="Uploaded Ultrasound Scan" style={{ maxWidth: '100%', marginTop: '20px', borderRadius: '8px' }} />
        </div>
      )}
    </div>
  );
};

export default ImageUpload;