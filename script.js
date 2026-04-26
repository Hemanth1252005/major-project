document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewSection = document.getElementById('preview-section');
    const imagePreview = document.getElementById('image-preview');
    const loader = document.getElementById('loader');
    const captionBox = document.getElementById('caption-box');
    const captionText = document.getElementById('caption-text');
    const resetBtn = document.getElementById('reset-btn');

    // Handle browse button click
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle drag and drop events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        });
    });

    // Handle file drop
    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const file = dt.files[0];
        handleFile(file);
    });

    // Handle file selection
    fileInput.addEventListener('change', function () {
        if (this.files && this.files[0]) {
            handleFile(this.files[0]);
        }
    });

    // Handle reset button
    resetBtn.addEventListener('click', () => {
        previewSection.classList.add('hidden');
        dropZone.classList.remove('hidden');
        captionBox.classList.add('hidden');
        fileInput.value = ''; // Reset input
    });

    function handleFile(file) {
        const validExtensions = ['jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'jfif', 'heic', 'tiff'];
        const fileExtension = file.name.split('.').pop().toLowerCase();
        const isImage = file.type.startsWith('image/') || validExtensions.includes(fileExtension);

        if (!isImage) {
            alert('Please upload an image file (JPEG, PNG, WEBP, etc).');
            return;
        }

        // Show preview quickly
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewSection.classList.remove('hidden');
            loader.classList.remove('hidden');
            captionBox.classList.add('hidden');

            // Upload to server
            uploadImage(file);
        };
        reader.readAsDataURL(file);
    }

    async function uploadImage(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                showError(data.error);
                return;
            }

            // Show result
            loader.classList.add('hidden');
            captionText.textContent = capitalizeFirstLetter(data.caption) + '.';
            captionBox.classList.remove('hidden');

        } catch (error) {
            console.error('Error:', error);
            showError('An error occurred during prediction.');
        }
    }

    function showError(message) {
        loader.classList.add('hidden');
        captionText.textContent = 'Error: ' + message;
        captionText.style.color = '#ef4444'; // Red color for error
        captionBox.classList.remove('hidden');
    }

    function capitalizeFirstLetter(string) {
        if (!string) return string;
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
});
