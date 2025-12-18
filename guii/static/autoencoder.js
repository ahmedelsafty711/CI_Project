
document.addEventListener('DOMContentLoaded', () => {
    // --- Canvas Setup ---
    const inputCanvas = document.getElementById('input-canvas');
    const outputCanvas = document.getElementById('output-canvas');
    const inputCtx = inputCanvas.getContext('2d');
    const outputCtx = outputCanvas.getContext('2d');

    const networkVis = document.getElementById('network-abstract-vis');
    const reconstructBtn = document.getElementById('reconstruct-btn');
    const clearBtn = document.getElementById('clear-btn');

    let isDrawing = false;
    
    const clearCanvas = () => {
        inputCtx.fillStyle = 'black';
        inputCtx.fillRect(0, 0, inputCanvas.width, inputCanvas.height);
        outputCtx.fillStyle = 'black';
        outputCtx.fillRect(0, 0, outputCanvas.width, outputCanvas.height);
    };

    const startDrawing = (e) => {
        isDrawing = true;
        draw(e);
    };

    const stopDrawing = () => {
        isDrawing = false;
        inputCtx.beginPath();
    };

    const draw = (e) => {
        if (!isDrawing) return;

        const rect = inputCanvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        inputCtx.lineWidth = 20;
        inputCtx.lineCap = 'round';
        inputCtx.strokeStyle = 'white';

        inputCtx.lineTo(x, y);
        inputCtx.stroke();
        inputCtx.beginPath();
        inputCtx.moveTo(x, y);
    };
    
    clearCanvas(); // Initial clear

    // --- Event Listeners ---
    inputCanvas.addEventListener('mousedown', startDrawing);
    inputCanvas.addEventListener('mouseup', stopDrawing);
    inputCanvas.addEventListener('mousemove', draw);
    inputCanvas.addEventListener('mouseout', stopDrawing);

    clearBtn.addEventListener('click', clearCanvas);
    reconstructBtn.addEventListener('click', () => {
        const imageData = getImageData();
        getReconstruction(imageData);
    });

    // --- Data & API ---
    const getImageData = () => {
        // 1. Downscale the 280x280 drawing to a 28x28 temporary canvas
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        tempCtx.drawImage(inputCanvas, 0, 0, 28, 28);
    
        // 2. Get pixel data and find the center of mass
        const imgData = tempCtx.getImageData(0, 0, 28, 28);
        const data = imgData.data;
        let sumX = 0, sumY = 0, count = 0;
    
        for (let y = 0; y < 28; y++) {
            for (let x = 0; x < 28; x++) {
                // Check the Red channel for a drawn pixel (since we draw in white)
                if (data[(y * 28 + x) * 4] > 0) { 
                    sumX += x;
                    sumY += y;
                    count++;
                }
            }
        }
        
        // If canvas is blank, return an empty array
        if (count === 0) {
            return new Array(784).fill(0);
        }
    
        // 3. Calculate the offset needed to center the digit
        const comX = sumX / count;
        const comY = sumY / count;
        const offsetX = 14 - comX; // 14 is the center of a 28px grid
        const offsetY = 14 - comY;
    
        // 4. Create a new, centered canvas
        const centeredCanvas = document.createElement('canvas');
        centeredCanvas.width = 28;
        centeredCanvas.height = 28;
        const centeredCtx = centeredCanvas.getContext('2d');
        centeredCtx.fillStyle = 'black';
        centeredCtx.fillRect(0, 0, 28, 28);
    
        // 5. Draw the original (downscaled) image onto the new canvas with the offset
        centeredCtx.drawImage(tempCanvas, offsetX, offsetY);
    
        // 6. Get the final pixel data from the centered canvas
        const finalImageData = centeredCtx.getImageData(0, 0, 28, 28);
        const pixelData = [];
        for (let i = 0; i < finalImageData.data.length; i += 4) {
            pixelData.push(finalImageData.data[i] / 255);
        }
    
        return pixelData;
    };

    const drawReconstruction = (pixelData) => {
        if (pixelData.length !== 784) return;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = 28;
        tempCanvas.height = 28;
        const tempCtx = tempCanvas.getContext('2d');
        const imageData = tempCtx.createImageData(28, 28);
        
        for (let i = 0; i < pixelData.length; i++) {
            const value = pixelData[i] * 255;
            imageData.data[i * 4] = value;     // R
            imageData.data[i * 4 + 1] = value; // G
            imageData.data[i * 4 + 2] = value; // B
            imageData.data[i * 4 + 3] = 255;   // Alpha
        }
        tempCtx.putImageData(imageData, 0, 0);

        // Scale up to the display canvas
        outputCtx.imageSmoothingEnabled = false; // Keep pixels sharp
        outputCtx.drawImage(tempCanvas, 0, 0, outputCanvas.width, outputCanvas.height);
    };

    const getReconstruction = async (imageData) => {
        try {
            const response = await fetch('/api/autoencoder/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input: imageData })
            });
            if (!response.ok) throw new Error('Prediction request failed');
            const data = await response.json();
            drawReconstruction(data.reconstruction);
        } catch (error) {
            console.error('Reconstruction Error:', error);
        }
    };

    // --- Network Visualization ---
    const drawAbstractNetwork = (layers) => {
        networkVis.innerHTML = '';
        layers.forEach((layer, index) => {
            const layerDiv = document.createElement('div');
            layerDiv.className = 'abstract-layer';
            
            const type = layer.type;
            const inputSize = layer.input_size;
            const outputSize = layer.output_size;
            let title = type;
            let desc = '';

            if (type === "Dense") {
                desc = `${inputSize} → ${outputSize}`;
            } else {
                title = `${type} Act.`;
            }

            layerDiv.innerHTML = `<h4>${title}</h4><p>${desc}</p>`;
            networkVis.appendChild(layerDiv);

            if (index < layers.length - 1) {
                const connectionsDiv = document.createElement('div');
                connectionsDiv.className = 'connections';
                networkVis.appendChild(connectionsDiv);
            }
        });
    };

    const initialize = async () => {
        try {
            const response = await fetch('/api/autoencoder/structure');
            if (!response.ok) throw new Error('Failed to fetch network structure');
            const data = await response.json();
            drawAbstractNetwork(data.layers);
        } catch (error) {
            console.error('Initialization Error:', error);
        }
    };

    initialize();
});
